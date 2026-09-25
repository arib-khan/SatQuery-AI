"""SatQuery — Gradio App Module

Exposes the Gradio Demo for modular imports.
"""

from app import get_demo

demo = None

if __name__ == "__main__":
    import os
    demo = get_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        ssr_mode=False,
    )
