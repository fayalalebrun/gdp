use std::cell::Cell;

use egui_glium::EguiGlium;
use glium::{
    glutin::{self, event_loop::ControlFlow, platform::run_return::EventLoopExtRunReturn},
    Display, Surface,
};

pub fn run(
    system: &mut super::State,
    display: Display,
    mut event_loop: glutin::event_loop::EventLoop<()>,
) {
    let draw_gui = Cell::new(true);
    let mut egui_glium = egui_glium::EguiGlium::new(&display, &event_loop);

    event_loop.run_return(move |event, _, control_flow| {
        let mut redraw = |display: &Display,
                          egui_glium: &mut EguiGlium,
                          _control_flow: &mut ControlFlow,
                          draw_gui: bool| {
            {
                egui_glium.run(&display, |ctx| system.paint_gui(ctx));
                let mut target = display.draw();
                target.clear_color(0.0, 0.0, 0.0, 0.0);
                target.clear_depth(1.0);
                system.draw(&mut target, display);

                if draw_gui {
                    egui_glium.paint(&display, &mut target);
                }
                target.finish().unwrap();
            }
        };

        match event {
            // Platform-dependent event handlers to workaround a winit bug
            // See: https://github.com/rust-windowing/winit/issues/987
            // See: https://github.com/rust-windowing/winit/issues/1619
            glutin::event::Event::RedrawEventsCleared if cfg!(windows) => {
                redraw(&display, &mut egui_glium, control_flow, draw_gui.get())
            }
            glutin::event::Event::RedrawRequested(_) if !cfg!(windows) => {
                redraw(&display, &mut egui_glium, control_flow, draw_gui.get())
            }
            glutin::event::Event::MainEventsCleared => {
                display.gl_window().window().request_redraw();
            }

            glutin::event::Event::WindowEvent { event, .. } => {
                let response = egui_glium.on_event(&event);

                display.gl_window().window().request_redraw();

                use glutin::event::WindowEvent;
                match &event {
                    WindowEvent::KeyboardInput {
                        input:
                            glutin::event::KeyboardInput {
                                virtual_keycode: Some(glutin::event::VirtualKeyCode::G),
                                state: glutin::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        draw_gui.set(!draw_gui.get());
                    }
                    event => {
                        if !response.consumed {
                            system.handle_window_event(&event)
                        }
                    }
                }
                if matches!(event, WindowEvent::CloseRequested | WindowEvent::Destroyed) {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                }
            }

            _ => (),
        }
    });
}

pub fn create_display(event_loop: &glutin::event_loop::EventLoop<()>) -> glium::Display {
    let window_builder = glutin::window::WindowBuilder::new().with_resizable(true);

    let context_builder = glutin::ContextBuilder::new()
        .with_depth_buffer(24)
        .with_srgb(true)
        .with_vsync(true)
        .with_gl_profile(glutin::GlProfile::Core)
        .with_gl(glutin::GlRequest::Latest);

    glium::Display::new(window_builder, context_builder, event_loop).unwrap()
}
