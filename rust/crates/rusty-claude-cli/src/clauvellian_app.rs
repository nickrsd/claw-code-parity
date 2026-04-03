use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::Duration;

use api::ProviderKind;
use eframe::egui::{
    self, Align, Color32, Context, CornerRadius, Frame, Key, LayerId, Layout, Margin, Pos2, Rect,
    RichText, ScrollArea, Stroke, TextEdit, Ui, Vec2,
};
use runtime::{ConfigLoader, PermissionMode, Session, TokenUsage};
use serde_json::json;

use crate::{
    build_runtime_with_events, build_system_prompt, create_managed_session_handle,
    final_assistant_text, format_auto_compaction_notice, list_managed_sessions, status_context,
    AllowedToolSet, InternalPromptProgressReporter, ManagedSessionSummary, SessionHandle,
};

pub(crate) type ClauvellianEventSink = Sender<ClauvellianBridgeEvent>;

#[derive(Debug, Clone)]
pub(crate) struct ClauvellianLaunchConfig {
    pub(crate) model: String,
    pub(crate) allowed_tools: Option<AllowedToolSet>,
    pub(crate) permission_mode: PermissionMode,
}

#[derive(Debug, Clone)]
pub(crate) enum ClauvellianBridgeEvent {
    TurnStarted {
        prompt: String,
    },
    ProgressLine {
        line: String,
    },
    TextDelta {
        delta: String,
    },
    ToolCallStart {
        name: String,
        input: String,
    },
    ToolCallResult {
        name: String,
        output: String,
        is_error: bool,
    },
    Usage {
        usage: TokenUsage,
    },
    AutoCompaction {
        notice: String,
    },
    TurnCompleted {
        final_text: String,
        usage: TokenUsage,
        iterations: usize,
    },
    TurnFailed {
        error: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
struct ChatEntry {
    role: ChatRole,
    content: String,
    pending: bool,
}

#[derive(Debug, Clone)]
struct ActivityEntry {
    title: String,
    detail: Option<String>,
    tone: ActivityTone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivityTone {
    Neutral,
    Accent,
    Success,
    Danger,
}

#[derive(Debug, Clone)]
struct ToolEntry {
    name: String,
    input: String,
    output: Option<String>,
    is_error: bool,
    completed: bool,
}

#[derive(Debug, Clone, Default)]
struct UsageSnapshot {
    input_tokens: u32,
    output_tokens: u32,
    cache_creation_input_tokens: u32,
    cache_read_input_tokens: u32,
    iterations: usize,
}

impl UsageSnapshot {
    fn update(&mut self, usage: &TokenUsage) {
        self.input_tokens = usage.input_tokens;
        self.output_tokens = usage.output_tokens;
        self.cache_creation_input_tokens = usage.cache_creation_input_tokens;
        self.cache_read_input_tokens = usage.cache_read_input_tokens;
    }
}

#[derive(Debug, Clone)]
struct WorkspaceSnapshot {
    cwd: String,
    branch: String,
    workspace_summary: String,
}

#[derive(Debug, Clone)]
struct ProviderSettings {
    path: PathBuf,
    openai_api_key: String,
    openai_base_url: String,
    status_message: Option<String>,
}

#[derive(Debug)]
struct ClauvellianApp {
    config: ClauvellianLaunchConfig,
    settings: ProviderSettings,
    session: SessionHandle,
    workspace: WorkspaceSnapshot,
    recent_sessions: Vec<ManagedSessionSummary>,
    events: Receiver<ClauvellianBridgeEvent>,
    composer: String,
    chat: Vec<ChatEntry>,
    activity: Vec<ActivityEntry>,
    tools: Vec<ToolEntry>,
    usage: UsageSnapshot,
    latest_status: String,
    pending: bool,
    should_scroll: bool,
    error_banner: Option<String>,
}

pub(crate) fn launch_clauvellian_app(
    config: ClauvellianLaunchConfig,
) -> Result<(), Box<dyn Error>> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(Vec2::new(1600.0, 980.0))
            .with_min_inner_size(Vec2::new(1180.0, 760.0))
            .with_title("Clauvellian"),
        ..Default::default()
    };

    eframe::run_native(
        "Clauvellian",
        native_options,
        Box::new(move |cc| match ClauvellianApp::new(cc, config.clone()) {
            Ok(app) => Ok(Box::new(app)),
            Err(error) => Err(Box::new(std::io::Error::other(error.to_string()))),
        }),
    )?;

    Ok(())
}

impl ClauvellianApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        mut config: ClauvellianLaunchConfig,
    ) -> Result<Self, Box<dyn Error>> {
        configure_theme(&cc.egui_ctx);
        if config.model == crate::DEFAULT_MODEL {
            config.model = "gpt-5.4".to_string();
        }

        let settings = ProviderSettings::load()?;
        settings.apply_to_env();
        let session = create_empty_session()?;
        let workspace = capture_workspace_snapshot()?;
        let recent_sessions = load_recent_sessions();
        let (_tx, rx_placeholder) = std::sync::mpsc::channel();

        Ok(Self {
            config,
            settings,
            session,
            workspace,
            recent_sessions,
            events: rx_placeholder,
            composer: String::new(),
            chat: Vec::new(),
            activity: vec![ActivityEntry {
                title: "Clauvellian is standing by".to_string(),
                detail: Some(
                    "The runtime is live behind the glass. Prompt it to inspect, plan, or change the workspace."
                        .to_string(),
                ),
                tone: ActivityTone::Accent,
            }],
            tools: Vec::new(),
            usage: UsageSnapshot::default(),
            latest_status: "Idle and waiting for a prompt".to_string(),
            pending: false,
            should_scroll: false,
            error_banner: None,
        })
    }

    fn submit_prompt(&mut self, ctx: &Context) {
        let prompt = self.composer.trim().to_string();
        if prompt.is_empty() || self.pending || !self.prompt_ready() {
            return;
        }

        let user_prompt = std::mem::take(&mut self.composer);
        self.chat.push(ChatEntry {
            role: ChatRole::User,
            content: user_prompt.trim().to_string(),
            pending: false,
        });
        self.chat.push(ChatEntry {
            role: ChatRole::Assistant,
            content: String::new(),
            pending: true,
        });
        self.pending = true;
        self.tools.clear();
        self.latest_status = "Analyzing request".to_string();
        self.should_scroll = true;
        self.error_banner = None;

        let (event_tx, event_rx) = std::sync::mpsc::channel();
        self.events = event_rx;

        let config = self.config.clone();
        let session = self.session.clone();
        thread::spawn(move || run_clauvellian_turn(config, session, prompt, event_tx));

        ctx.request_repaint();
    }

    fn start_new_session(&mut self) {
        if self.pending {
            return;
        }

        match create_empty_session() {
            Ok(session) => {
                self.session = session;
                self.chat.clear();
                self.tools.clear();
                self.push_activity(ActivityEntry {
                    title: "Started a fresh session".to_string(),
                    detail: Some(
                        "The new transcript is empty, but the recent rail still keeps the older sessions within reach."
                            .to_string(),
                    ),
                    tone: ActivityTone::Success,
                });
                self.latest_status = "Fresh session created".to_string();
                self.error_banner = None;
                self.usage = UsageSnapshot::default();
                self.refresh_recent_sessions();
            }
            Err(error) => {
                self.error_banner = Some(error.to_string());
            }
        }
    }

    fn refresh_recent_sessions(&mut self) {
        self.recent_sessions = load_recent_sessions();
    }

    fn poll_runtime(&mut self, ctx: &Context) {
        while let Ok(event) = self.events.try_recv() {
            self.handle_event(event);
        }

        ctx.request_repaint_after(Duration::from_millis(33));
    }

    fn handle_event(&mut self, event: ClauvellianBridgeEvent) {
        match event {
            ClauvellianBridgeEvent::TurnStarted { prompt } => {
                self.push_activity(ActivityEntry {
                    title: "Prompt queued".to_string(),
                    detail: Some(prompt),
                    tone: ActivityTone::Neutral,
                });
                self.should_scroll = true;
            }
            ClauvellianBridgeEvent::ProgressLine { line } => {
                self.latest_status = line.clone();
                self.push_activity(ActivityEntry {
                    title: "Working trace".to_string(),
                    detail: Some(line),
                    tone: ActivityTone::Accent,
                });
            }
            ClauvellianBridgeEvent::TextDelta { delta } => {
                if let Some(last) = self.chat.last_mut() {
                    last.content.push_str(&delta);
                }
                self.should_scroll = true;
            }
            ClauvellianBridgeEvent::ToolCallStart { name, input } => {
                self.tools.push(ToolEntry {
                    name: name.clone(),
                    input: input.clone(),
                    output: None,
                    is_error: false,
                    completed: false,
                });
                self.trim_tools();
                self.push_activity(ActivityEntry {
                    title: format!("Running {name}"),
                    detail: Some(truncate_for_panel(&input, 180)),
                    tone: ActivityTone::Neutral,
                });
                self.should_scroll = true;
            }
            ClauvellianBridgeEvent::ToolCallResult {
                name,
                output,
                is_error,
            } => {
                if let Some(tool) = self
                    .tools
                    .iter_mut()
                    .rev()
                    .find(|tool| tool.name == name && !tool.completed)
                {
                    tool.output = Some(output.clone());
                    tool.completed = true;
                    tool.is_error = is_error;
                }
                self.push_activity(ActivityEntry {
                    title: if is_error {
                        format!("{name} failed")
                    } else {
                        format!("{name} finished")
                    },
                    detail: Some(truncate_for_panel(&output, 180)),
                    tone: if is_error {
                        ActivityTone::Danger
                    } else {
                        ActivityTone::Success
                    },
                });
            }
            ClauvellianBridgeEvent::Usage { usage } => {
                self.usage.update(&usage);
            }
            ClauvellianBridgeEvent::AutoCompaction { notice } => {
                self.push_activity(ActivityEntry {
                    title: "Session compacted".to_string(),
                    detail: Some(notice),
                    tone: ActivityTone::Accent,
                });
            }
            ClauvellianBridgeEvent::TurnCompleted {
                final_text,
                usage,
                iterations,
            } => {
                self.pending = false;
                self.latest_status = "Turn complete".to_string();
                self.usage.update(&usage);
                self.usage.iterations = iterations;
                if let Some(last) = self.chat.last_mut() {
                    last.pending = false;
                    if last.content.trim().is_empty() {
                        last.content = final_text;
                    }
                }
                self.push_activity(ActivityEntry {
                    title: "Response complete".to_string(),
                    detail: Some(format!(
                        "{iterations} iterations | {} input | {} output tokens",
                        usage.input_tokens, usage.output_tokens
                    )),
                    tone: ActivityTone::Success,
                });
                self.refresh_recent_sessions();
                self.should_scroll = true;
            }
            ClauvellianBridgeEvent::TurnFailed { error } => {
                self.pending = false;
                self.latest_status = "Turn failed".to_string();
                self.error_banner = Some(error.clone());
                if let Some(last) = self.chat.last_mut() {
                    last.pending = false;
                    if last.content.trim().is_empty() {
                        last.content = format!("Request failed.\n\n{error}");
                    }
                }
                self.push_activity(ActivityEntry {
                    title: "Request failed".to_string(),
                    detail: Some(error),
                    tone: ActivityTone::Danger,
                });
            }
        }
    }

    fn push_activity(&mut self, entry: ActivityEntry) {
        self.activity.push(entry);
        if self.activity.len() > 80 {
            let remove_count = self.activity.len() - 80;
            self.activity.drain(0..remove_count);
        }
    }

    fn trim_tools(&mut self) {
        if self.tools.len() > 24 {
            let remove_count = self.tools.len() - 24;
            self.tools.drain(0..remove_count);
        }
    }

    fn render_top_bar(&mut self, ctx: &Context) {
        egui::TopBottomPanel::top("clauvellian_top_bar")
            .frame(glass_frame(Color32::from_rgba_premultiplied(13, 18, 24, 214)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new("Clauvellian")
                                .size(28.0)
                                .strong()
                                .color(accent_orange()),
                        );
                        ui.label(
                            RichText::new(
                                "A CRT-lit coding deck for the Claw runtime, borrowing the command-center feel of Codex and Claude without losing the terminal soul.",
                            )
                            .color(muted_text()),
                        );
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let button = egui::Button::new("New Session")
                            .fill(Color32::from_rgba_premultiplied(238, 128, 76, 42))
                            .stroke(Stroke::new(1.0, accent_orange()))
                            .min_size(Vec2::new(132.0, 40.0));
                        if ui.add_enabled(!self.pending, button).clicked() {
                            self.start_new_session();
                        }
                        ui.add_space(12.0);
                        if self.requires_openai_key() && !self.settings.has_openai_key() {
                            status_chip(ui, "OpenAI key required", accent_purple());
                        }
                        status_chip(
                            ui,
                            &self.config.permission_mode.as_str().replace('-', " "),
                            accent_green(),
                        );
                        status_chip(ui, &self.config.model, accent_purple());
                    });
                });
            });
    }

    fn render_left_rail(&mut self, ctx: &Context) {
        egui::SidePanel::left("clauvellian_left_rail")
            .resizable(true)
            .default_width(300.0)
            .frame(glass_frame(Color32::from_rgba_premultiplied(10, 14, 18, 178)))
            .show(ctx, |ui| {
                ui.heading(RichText::new("Session").size(18.0).color(bright_text()));
                glass_card(ui, Color32::from_rgba_premultiplied(18, 24, 31, 176), |ui| {
                    metric_line(ui, "ID", &self.session.id);
                    metric_line(ui, "Branch", &self.workspace.branch);
                    metric_line(ui, "Scope", &self.workspace.workspace_summary);
                    metric_line(ui, "Path", &self.session.path.display().to_string());
                });

                ui.add_space(14.0);
                ui.heading(RichText::new("Workspace").size(18.0).color(bright_text()));
                glass_card(ui, Color32::from_rgba_premultiplied(17, 24, 29, 166), |ui| {
                    ui.label(
                        RichText::new(&self.workspace.cwd)
                            .monospace()
                            .color(secondary_text()),
                    );
                });

                ui.add_space(14.0);
                ui.heading(RichText::new("Provider").size(18.0).color(bright_text()));
                self.render_provider_settings(ui);

                ui.add_space(14.0);
                ui.heading(
                    RichText::new("Recent Sessions")
                        .size(18.0)
                        .color(bright_text()),
                );
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.recent_sessions.is_empty() {
                            ui.label(
                                RichText::new("Completed sessions will settle here as the transcript history grows.")
                                    .color(muted_text()),
                            );
                        }
                        for session in self.recent_sessions.iter().take(10) {
                            let is_active = session.id == self.session.id;
                            let fill = if is_active {
                                Color32::from_rgba_premultiplied(30, 48, 60, 205)
                            } else {
                                Color32::from_rgba_premultiplied(15, 22, 29, 172)
                            };
                            let stroke = if is_active {
                                Stroke::new(1.0, accent_orange().gamma_multiply(0.8))
                            } else {
                                Stroke::new(1.0, subtle_border())
                            };
                            Frame::new()
                                .fill(fill)
                                .stroke(stroke)
                                .corner_radius(CornerRadius::same(14))
                                .inner_margin(Margin::same(12))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(if is_active { "Active" } else { "Saved" })
                                                .strong()
                                                .color(if is_active {
                                                    accent_orange()
                                                } else {
                                                    muted_text()
                                                }),
                                        );
                                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                            ui.label(
                                                RichText::new(format!("{} msgs", session.message_count))
                                                    .color(muted_text()),
                                            );
                                        });
                                    });
                                    ui.label(
                                        RichText::new(&session.id)
                                            .monospace()
                                            .color(bright_text()),
                                    );
                                    if let Some(branch_name) = &session.branch_name {
                                        ui.label(
                                            RichText::new(branch_name)
                                                .color(secondary_text())
                                                .italics(),
                                        );
                                    }
                                });
                            ui.add_space(8.0);
                        }
                    });
            });
    }

    fn render_activity_panel(&mut self, ctx: &Context) {
        egui::SidePanel::right("clauvellian_activity_panel")
            .resizable(true)
            .default_width(380.0)
            .frame(glass_frame(Color32::from_rgba_premultiplied(8, 13, 18, 186)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(
                        RichText::new("Working Trace")
                            .size(18.0)
                            .color(bright_text()),
                    );
                    if self.pending {
                        ui.label(RichText::new("live").strong().color(accent_orange()));
                    }
                });
                ui.label(RichText::new(&self.latest_status).color(muted_text()).italics());

                ui.add_space(12.0);
                glass_card(ui, Color32::from_rgba_premultiplied(16, 25, 31, 182), |ui| {
                    metric_row(ui, "Input", &self.usage.input_tokens.to_string());
                    metric_row(ui, "Output", &self.usage.output_tokens.to_string());
                    metric_row(
                        ui,
                        "Cache Build",
                        &self.usage.cache_creation_input_tokens.to_string(),
                    );
                    metric_row(
                        ui,
                        "Cache Read",
                        &self.usage.cache_read_input_tokens.to_string(),
                    );
                    metric_row(ui, "Iterations", &self.usage.iterations.to_string());
                });

                ui.add_space(12.0);
                ui.heading(
                    RichText::new("Tool Activity")
                        .size(18.0)
                        .color(bright_text()),
                );
                ScrollArea::vertical().max_height(250.0).show(ui, |ui| {
                    if self.tools.is_empty() {
                        ui.label(
                            RichText::new(
                                "Tool calls will surface here once the runtime reaches out into the workspace.",
                            )
                            .color(muted_text()),
                        );
                    }
                    for tool in self.tools.iter().rev() {
                        let accent = if tool.is_error {
                            accent_red()
                        } else if tool.completed {
                            accent_green()
                        } else {
                            accent_purple()
                        };
                        Frame::new()
                            .fill(Color32::from_rgba_premultiplied(17, 23, 30, 176))
                            .stroke(Stroke::new(1.0, accent.gamma_multiply(0.6)))
                            .corner_radius(CornerRadius::same(14))
                            .inner_margin(Margin::same(10))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(&tool.name)
                                            .strong()
                                            .color(bright_text()),
                                    );
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        let label = if tool.completed {
                                            if tool.is_error { "error" } else { "done" }
                                        } else {
                                            "running"
                                        };
                                        ui.label(RichText::new(label).strong().color(accent));
                                    });
                                });
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(truncate_for_panel(&tool.input, 180))
                                        .monospace()
                                        .color(secondary_text()),
                                );
                                if let Some(output) = &tool.output {
                                    egui::CollapsingHeader::new("Output")
                                        .default_open(tool.is_error)
                                        .show(ui, |ui| {
                                            ui.label(
                                                RichText::new(output)
                                                    .monospace()
                                                    .color(bright_text()),
                                            );
                                        });
                                }
                            });
                        ui.add_space(8.0);
                    }
                });

                ui.add_space(12.0);
                ui.heading(RichText::new("Activity Feed").size(18.0).color(bright_text()));
                ScrollArea::vertical().show(ui, |ui| {
                    for item in self.activity.iter().rev() {
                        glass_card(ui, Color32::from_rgba_premultiplied(16, 24, 30, 158), |ui| {
                            ui.label(
                                RichText::new(&item.title)
                                    .strong()
                                    .color(tone_color(item.tone)),
                            );
                            if let Some(detail) = &item.detail {
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(detail)
                                        .color(secondary_text())
                                        .monospace(),
                                );
                            }
                        });
                        ui.add_space(8.0);
                    }
                });
            });
    }

    fn render_conversation(&mut self, ctx: &Context) {
        egui::CentralPanel::default()
            .frame(
                Frame::new()
                    .fill(Color32::TRANSPARENT)
                    .inner_margin(Margin::same(22)),
            )
            .show(ctx, |ui| {
                if !self.prompt_ready() {
                    welcome_surface(ui, &mut self.composer);
                    ui.add_space(18.0);
                    self.render_provider_gate(ui);
                    return;
                }

                if let Some(error) = &self.error_banner {
                    Frame::new()
                        .fill(Color32::from_rgba_premultiplied(72, 24, 25, 214))
                        .stroke(Stroke::new(1.0, accent_red()))
                        .corner_radius(CornerRadius::same(14))
                        .inner_margin(Margin::same(12))
                        .show(ui, |ui| {
                            ui.label(RichText::new(error).color(Color32::WHITE).strong());
                        });
                    ui.add_space(14.0);
                }

                if self.chat.is_empty() {
                    welcome_surface(ui, &mut self.composer);
                    return;
                }

                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(self.should_scroll)
                    .show(ui, |ui| {
                        for entry in &self.chat {
                            render_chat_entry(ui, entry);
                            ui.add_space(14.0);
                        }
                        if self.should_scroll {
                            ui.scroll_to_cursor(Some(Align::BOTTOM));
                            self.should_scroll = false;
                        }
                    });
            });
    }

    fn render_composer(&mut self, ctx: &Context) {
        egui::TopBottomPanel::bottom("clauvellian_composer")
            .frame(glass_frame(Color32::from_rgba_premultiplied(10, 15, 20, 214)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Prompt Deck")
                            .size(17.0)
                            .strong()
                            .color(bright_text()),
                    );
                    if self.pending {
                        ui.label(
                            RichText::new("streaming trace")
                                .strong()
                                .color(accent_purple()),
                        );
                    }
                });
                ui.add_space(10.0);

                let input = TextEdit::multiline(&mut self.composer)
                    .desired_rows(4)
                    .hint_text("Ask Clauvellian to inspect files, explain architecture, plan a change, or execute it.")
                    .lock_focus(true)
                    .desired_width(f32::INFINITY);
                let response = ui.add_enabled(!self.pending, input);

                if response.has_focus()
                    && ui.input(|input| input.key_pressed(Key::Enter) && input.modifiers.command)
                {
                    self.submit_prompt(ctx);
                }

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Cmd/Ctrl+Enter sends | Enter adds a newline | Live trace stays visible in the right rail")
                            .color(muted_text()),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let button = egui::Button::new(
                            RichText::new(if self.pending { "Working..." } else { "Send Prompt" })
                                .strong(),
                        )
                        .fill(Color32::from_rgba_premultiplied(238, 128, 76, 46))
                        .stroke(Stroke::new(1.0, accent_orange()))
                        .min_size(Vec2::new(150.0, 42.0));
                        if ui
                            .add_enabled(
                                !self.pending
                                    && self.prompt_ready()
                                    && !self.composer.trim().is_empty(),
                                button,
                            )
                            .clicked()
                        {
                            self.submit_prompt(ctx);
                        }
                    });
                });
            });
    }

    fn requires_openai_key(&self) -> bool {
        matches!(
            api::detect_provider_kind(&self.config.model),
            ProviderKind::OpenAi
        )
    }

    fn prompt_ready(&self) -> bool {
        if self.requires_openai_key() {
            self.settings.has_openai_key()
        } else {
            true
        }
    }
}

impl eframe::App for ClauvellianApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.poll_runtime(ctx);
        paint_crt_backdrop(ctx);
        self.render_top_bar(ctx);
        self.render_left_rail(ctx);
        self.render_activity_panel(ctx);
        self.render_conversation(ctx);
        self.render_composer(ctx);
    }
}

fn run_clauvellian_turn(
    config: ClauvellianLaunchConfig,
    session: SessionHandle,
    prompt: String,
    event_sink: ClauvellianEventSink,
) {
    let _ = event_sink.send(ClauvellianBridgeEvent::TurnStarted {
        prompt: prompt.clone(),
    });

    let result = (|| -> Result<(String, TokenUsage, usize, Option<String>), Box<dyn Error>> {
        let system_prompt = build_system_prompt()?;
        let base_session = Session::load_from_path(&session.path)
            .unwrap_or_else(|_| Session::new().with_persistence_path(session.path.clone()));
        let progress_reporter =
            InternalPromptProgressReporter::clauvellian(&prompt, event_sink.clone());
        let mut runtime = build_runtime_with_events(
            base_session,
            &session.id,
            config.model.clone(),
            system_prompt,
            true,
            false,
            config.allowed_tools.clone(),
            config.permission_mode,
            Some(progress_reporter),
            Some(event_sink.clone()),
        )?;
        let summary = runtime.run_turn(prompt, None)?;
        runtime.session().save_to_path(&session.path)?;
        let final_text = final_assistant_text(&summary);
        let usage = summary.usage.clone();
        let iterations = summary.iterations;
        let auto_notice = summary
            .auto_compaction
            .map(|event| format_auto_compaction_notice(event.removed_message_count));
        runtime.shutdown_plugins()?;
        Ok((final_text, usage, iterations, auto_notice))
    })();

    match result {
        Ok((final_text, usage, iterations, auto_notice)) => {
            if let Some(notice) = auto_notice {
                let _ = event_sink.send(ClauvellianBridgeEvent::AutoCompaction { notice });
            }
            let _ = event_sink.send(ClauvellianBridgeEvent::TurnCompleted {
                final_text,
                usage,
                iterations,
            });
        }
        Err(error) => {
            let _ = event_sink.send(ClauvellianBridgeEvent::TurnFailed {
                error: error.to_string(),
            });
        }
    }
}

fn create_empty_session() -> Result<SessionHandle, Box<dyn Error>> {
    let session = Session::new();
    let handle = create_managed_session_handle(&session.session_id)?;
    session
        .with_persistence_path(handle.path.clone())
        .save_to_path(&handle.path)?;
    Ok(handle)
}

fn capture_workspace_snapshot() -> Result<WorkspaceSnapshot, Box<dyn Error>> {
    let snapshot = status_context(None)?;
    Ok(WorkspaceSnapshot {
        cwd: snapshot.cwd.display().to_string(),
        branch: snapshot.git_branch.unwrap_or_else(|| "unknown".to_string()),
        workspace_summary: snapshot.git_summary.headline(),
    })
}

fn load_recent_sessions() -> Vec<ManagedSessionSummary> {
    list_managed_sessions().unwrap_or_default()
}

impl ProviderSettings {
    fn load() -> Result<Self, Box<dyn Error>> {
        let cwd = std::env::current_dir()?;
        let loader = ConfigLoader::default_for(&cwd);
        let path = loader
            .config_home()
            .join("clauvellian")
            .join("settings.json");

        let mut settings = Self {
            path,
            openai_api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            openai_base_url: std::env::var("OPENAI_BASE_URL").unwrap_or_default(),
            status_message: None,
        };

        if settings.path.exists() {
            let value: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&settings.path)?)?;
            if let Some(key) = value.get("openai_api_key").and_then(|value| value.as_str()) {
                if settings.openai_api_key.trim().is_empty() {
                    settings.openai_api_key = key.to_string();
                }
            }
            if let Some(base_url) = value
                .get("openai_base_url")
                .and_then(|value| value.as_str())
            {
                if settings.openai_base_url.trim().is_empty() {
                    settings.openai_base_url = base_url.to_string();
                }
            }
        }

        Ok(settings)
    }

    fn save(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let payload = json!({
            "openai_api_key": self.openai_api_key.trim(),
            "openai_base_url": self.openai_base_url.trim(),
        });
        fs::write(&self.path, serde_json::to_string_pretty(&payload)?)?;
        self.status_message = Some(format!("Saved OpenAI settings to {}", self.path.display()));
        Ok(())
    }

    fn apply_to_env(&self) {
        let api_key = self.openai_api_key.trim();
        if !api_key.is_empty() {
            std::env::set_var("OPENAI_API_KEY", api_key);
        }

        let base_url = self.openai_base_url.trim();
        if !base_url.is_empty() {
            std::env::set_var("OPENAI_BASE_URL", base_url);
        } else if !self.openai_api_key.trim().is_empty() {
            std::env::remove_var("OPENAI_BASE_URL");
        }
    }

    fn has_openai_key(&self) -> bool {
        !self.openai_api_key.trim().is_empty()
            || std::env::var("OPENAI_API_KEY")
                .map(|value| !value.trim().is_empty())
                .unwrap_or(false)
    }
}

impl ClauvellianApp {
    fn render_provider_settings(&mut self, ui: &mut Ui) {
        glass_card(
            ui,
            Color32::from_rgba_premultiplied(18, 20, 33, 178),
            |ui| {
                ui.label(RichText::new("Model").strong().color(accent_purple()));
                ui.add(
                    TextEdit::singleline(&mut self.config.model)
                        .desired_width(f32::INFINITY)
                        .hint_text("gpt-5.4 or codex-mini-latest"),
                );
                ui.add_space(10.0);
                ui.label(
                    RichText::new("OpenAI API Key")
                        .strong()
                        .color(accent_orange()),
                );
                ui.add(
                    TextEdit::singleline(&mut self.settings.openai_api_key)
                        .desired_width(f32::INFINITY)
                        .password(true)
                        .hint_text("sk-..."),
                );
                ui.add_space(10.0);
                ui.label(
                    RichText::new("OpenAI Base URL")
                        .strong()
                        .color(accent_purple()),
                );
                ui.add(
                    TextEdit::singleline(&mut self.settings.openai_base_url)
                        .desired_width(f32::INFINITY)
                        .hint_text("Optional: https://api.openai.com/v1"),
                );
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let apply_clicked = ui
                        .add(
                            egui::Button::new("Apply")
                                .fill(Color32::from_rgba_premultiplied(128, 82, 205, 36))
                                .stroke(Stroke::new(1.0, accent_purple())),
                        )
                        .clicked();
                    let save_clicked = ui
                        .add(
                            egui::Button::new("Save")
                                .fill(Color32::from_rgba_premultiplied(238, 128, 76, 40))
                                .stroke(Stroke::new(1.0, accent_orange())),
                        )
                        .clicked();

                    if apply_clicked {
                        self.settings.apply_to_env();
                        self.settings.status_message =
                            Some("Applied OpenAI settings to the current app session.".to_string());
                    }
                    if save_clicked {
                        match self.settings.save() {
                            Ok(()) => self.settings.apply_to_env(),
                            Err(error) => {
                                self.settings.status_message =
                                    Some(format!("Could not save settings: {error}"));
                            }
                        }
                    }
                });
                ui.add_space(8.0);
                ui.label(
                RichText::new("The key is used when the current model resolves to OpenAI-compatible routing.")
                    .color(muted_text()),
            );
                if let Some(message) = &self.settings.status_message {
                    ui.add_space(8.0);
                    ui.label(RichText::new(message).color(secondary_text()));
                }
            },
        );
    }

    fn render_provider_gate(&mut self, ui: &mut Ui) {
        glass_card(
            ui,
            Color32::from_rgba_premultiplied(28, 18, 35, 188),
            |ui| {
                ui.label(
                    RichText::new("OpenAI setup required")
                        .size(24.0)
                        .strong()
                        .color(accent_purple()),
                );
                ui.add_space(8.0);
                ui.label(
                RichText::new(
                    "This model path needs an OpenAI API key before the prompt deck can go live. Add it in the Provider card on the left, then hit Apply or Save.",
                )
                .color(bright_text()),
            );
            },
        );
    }
}

fn configure_theme(ctx: &Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(bright_text());
    visuals.window_fill = Color32::from_rgba_premultiplied(8, 12, 16, 170);
    visuals.panel_fill = Color32::TRANSPARENT;
    visuals.extreme_bg_color = Color32::from_rgb(5, 7, 11);
    visuals.faint_bg_color = Color32::from_rgba_premultiplied(16, 12, 20, 120);
    visuals.widgets.inactive.bg_fill = Color32::from_rgba_premultiplied(27, 20, 33, 138);
    visuals.widgets.hovered.bg_fill = Color32::from_rgba_premultiplied(40, 28, 48, 178);
    visuals.widgets.active.bg_fill = Color32::from_rgba_premultiplied(54, 36, 61, 210);
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, subtle_border());
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, accent_purple().gamma_multiply(0.7));
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, accent_orange().gamma_multiply(0.75));
    visuals.widgets.inactive.fg_stroke.color = bright_text();
    visuals.widgets.hovered.fg_stroke.color = Color32::WHITE;
    visuals.widgets.active.fg_stroke.color = Color32::WHITE;
    visuals.selection.bg_fill = accent_orange().gamma_multiply(0.7);
    visuals.window_stroke = Stroke::new(1.0, subtle_border());
    ctx.set_visuals(visuals);
}

fn paint_crt_backdrop(ctx: &Context) {
    let rect = ctx.screen_rect();
    let painter = ctx.layer_painter(LayerId::background());
    let time = ctx.input(|input| input.time) as f32;

    painter.rect_filled(rect, 0.0, Color32::from_rgb(5, 7, 11));

    let swirls = [
        (
            0.22_f32,
            0.24_f32,
            250.0_f32,
            120.0_f32,
            Color32::from_rgba_premultiplied(255, 128, 68, 24),
        ),
        (
            0.74_f32,
            0.30_f32,
            300.0_f32,
            140.0_f32,
            Color32::from_rgba_premultiplied(156, 92, 242, 24),
        ),
        (
            0.58_f32,
            0.76_f32,
            330.0_f32,
            160.0_f32,
            Color32::from_rgba_premultiplied(108, 54, 186, 20),
        ),
    ];

    for (x_ratio, y_ratio, radius, drift, color) in swirls {
        let center = Pos2::new(
            rect.left() + rect.width() * x_ratio + time.sin() * drift * 0.18,
            rect.top() + rect.height() * y_ratio + time.cos() * drift * 0.12,
        );
        for layer in (1..=4).rev() {
            let factor = layer as f32 / 4.0;
            let layered_color = Color32::from_rgba_premultiplied(
                color.r(),
                color.g(),
                color.b(),
                ((color.a() as f32) * factor) as u8,
            );
            painter.circle_filled(center, radius * factor, layered_color);
        }
    }

    let scanline = Color32::from_rgba_premultiplied(201, 181, 255, 8);
    let mut y = rect.top();
    while y < rect.bottom() {
        painter.line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            Stroke::new(1.0, scanline),
        );
        y += 4.0;
    }

    let flicker_strength = ((((time * 0.9).sin() * 0.5) + 0.5).powf(7.0) * 16.0) as u8;
    painter.rect_filled(
        rect,
        0.0,
        Color32::from_rgba_premultiplied(255, 255, 255, flicker_strength),
    );

    for index in 0..3 {
        let phase = time * (0.35 + index as f32 * 0.08) + index as f32 * 1.7;
        let band_y = rect.top() + rect.height() * (phase.sin() * 0.08 + 0.5);
        let band = Rect::from_min_max(
            Pos2::new(rect.left(), band_y),
            Pos2::new(rect.right(), band_y + 2.0 + index as f32),
        );
        let alpha = 6 + index as u8 * 2;
        painter.rect_filled(
            band,
            0.0,
            Color32::from_rgba_premultiplied(255, 255, 255, alpha),
        );
    }

    let vignette = [
        Rect::from_min_max(
            rect.min,
            Pos2::new(rect.right(), rect.top() + rect.height() * 0.12),
        ),
        Rect::from_min_max(
            Pos2::new(rect.left(), rect.bottom() - rect.height() * 0.12),
            rect.max,
        ),
        Rect::from_min_max(
            rect.min,
            Pos2::new(rect.left() + rect.width() * 0.08, rect.bottom()),
        ),
        Rect::from_min_max(
            Pos2::new(rect.right() - rect.width() * 0.08, rect.top()),
            rect.max,
        ),
    ];
    for band in vignette {
        painter.rect_filled(band, 0.0, Color32::from_rgba_premultiplied(0, 0, 0, 34));
    }
}

fn glass_frame(fill: Color32) -> Frame {
    Frame::new()
        .fill(fill)
        .stroke(Stroke::new(1.0, subtle_border()))
        .corner_radius(CornerRadius::same(18))
        .inner_margin(Margin::same(16))
}

fn glass_card(ui: &mut Ui, fill: Color32, add_contents: impl FnOnce(&mut Ui)) {
    Frame::new()
        .fill(fill)
        .stroke(Stroke::new(1.0, subtle_border()))
        .corner_radius(CornerRadius::same(16))
        .inner_margin(Margin::same(12))
        .show(ui, add_contents);
}

fn status_chip(ui: &mut Ui, text: &str, fill: Color32) {
    Frame::new()
        .fill(fill.gamma_multiply(0.14))
        .stroke(Stroke::new(1.0, fill.gamma_multiply(0.7)))
        .corner_radius(CornerRadius::same(255))
        .inner_margin(Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.label(RichText::new(text).color(fill).strong());
        });
}

fn welcome_surface(ui: &mut Ui, composer: &mut String) {
    ui.add_space(28.0);
    glass_card(
        ui,
        Color32::from_rgba_premultiplied(14, 12, 22, 194),
        |ui| {
            ui.label(
                RichText::new("Clauvellian")
                    .size(44.0)
                    .strong()
                    .color(accent_orange()),
            );
            ui.add_space(8.0);
            ui.label(
            RichText::new(
                "A cinematic coding cockpit with a live working trace, tool feed, session memory, and a dark CRT wash underneath the glass.",
            )
            .size(17.0)
            .color(bright_text()),
        );
            ui.add_space(16.0);
            ui.label(
                RichText::new("Starter prompts")
                    .strong()
                    .color(accent_purple()),
            );
            ui.add_space(8.0);

            let prompts = [
            "Audit this repo and tell me where the runtime still assumes Anthropic-specific behavior.",
            "Map the UI architecture and sketch the cleanest path to a Tauri shell.",
            "Review the last few commits and tell me what risks remain before we ship the desktop app.",
        ];

            for prompt in prompts {
                let button =
                    egui::Button::new(RichText::new(prompt).color(bright_text()).size(14.0))
                        .fill(Color32::from_rgba_premultiplied(28, 22, 38, 176))
                        .stroke(Stroke::new(1.0, subtle_border()))
                        .min_size(Vec2::new(ui.available_width(), 42.0));
                if ui.add(button).clicked() {
                    *composer = prompt.to_string();
                }
                ui.add_space(8.0);
            }
        },
    );
}

fn render_chat_entry(ui: &mut Ui, entry: &ChatEntry) {
    let is_user = entry.role == ChatRole::User;
    let lead = if is_user { "You" } else { "Clauvellian" };
    let accent = if is_user {
        accent_purple()
    } else {
        accent_orange()
    };
    let bubble_fill = if is_user {
        Color32::from_rgba_premultiplied(32, 22, 49, 200)
    } else {
        Color32::from_rgba_premultiplied(24, 18, 28, 196)
    };

    ui.horizontal(|ui| {
        let width = ui.available_width();
        let spacer = width * 0.16;
        if is_user {
            ui.add_space(spacer);
        }

        let max_width = (ui.available_width() - 8.0).max(320.0);
        let bubble = Frame::new()
            .fill(bubble_fill)
            .stroke(Stroke::new(1.0, accent.gamma_multiply(0.55)))
            .corner_radius(CornerRadius::same(18))
            .inner_margin(Margin::same(14));

        bubble.show(ui, |ui| {
            ui.set_max_width(max_width);
            ui.label(RichText::new(lead).strong().color(accent));
            ui.add_space(6.0);
            let body = if entry.pending && entry.content.trim().is_empty() {
                "Working through the request..."
            } else {
                &entry.content
            };
            ui.label(RichText::new(body).color(bright_text()));
        });

        if !is_user {
            ui.add_space(spacer);
        }
    });
}

fn metric_line(ui: &mut Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(muted_text()).strong());
        ui.add_space(10.0);
        ui.label(RichText::new(value).monospace().color(bright_text()));
    });
}

fn metric_row(ui: &mut Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(muted_text()));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(value).monospace().color(bright_text()));
        });
    });
}

fn truncate_for_panel(value: &str, limit: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let preview = chars.by_ref().take(limit).collect::<String>();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        compact
    }
}

fn tone_color(tone: ActivityTone) -> Color32 {
    match tone {
        ActivityTone::Neutral => bright_text(),
        ActivityTone::Accent => accent_purple(),
        ActivityTone::Success => accent_green(),
        ActivityTone::Danger => accent_red(),
    }
}

fn accent_orange() -> Color32 {
    Color32::from_rgb(238, 128, 76)
}

fn accent_purple() -> Color32 {
    Color32::from_rgb(164, 98, 245)
}

fn accent_green() -> Color32 {
    Color32::from_rgb(112, 196, 148)
}

fn accent_red() -> Color32 {
    Color32::from_rgb(220, 104, 101)
}

fn bright_text() -> Color32 {
    Color32::from_rgb(239, 238, 245)
}

fn secondary_text() -> Color32 {
    Color32::from_rgb(196, 188, 214)
}

fn muted_text() -> Color32 {
    Color32::from_rgb(145, 136, 161)
}

fn subtle_border() -> Color32 {
    Color32::from_rgba_premultiplied(155, 126, 204, 84)
}
