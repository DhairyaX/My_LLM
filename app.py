import gradio as gr
import spaces

from inference import MiniGPT


# -----------------------------
# Load Model Once
# -----------------------------

model = MiniGPT()


# -----------------------------
# Generation Function
# -----------------------------

@spaces.GPU
def generate_text(
    prompt,
    temperature,
    top_k,
    max_new_tokens
):

    if not prompt.strip():
        return "Please enter a prompt."

    try:
        output = model.generate(
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k)
        )

        # Remove the original prompt if model.generate()
        # returns the prompt together with the continuation.
        if output.startswith(prompt):
            output = output[len(prompt):]

        return output.strip()

    except KeyError as e:

        return (
            f"Character {e} is not in "
            "the model vocabulary."
        )

    except Exception as e:

        return f"Generation error: {str(e)}"


# -----------------------------
# Interface
# -----------------------------

demo = gr.Interface(

    fn=generate_text,

    inputs=[

        gr.Textbox(
            label="Prompt",
            placeholder="Enter a prompt...",
            value="the queen"
        ),

        gr.Slider(
            minimum=0.2,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Temperature"
        ),

        gr.Slider(
            minimum=1,
            maximum=50,
            value=20,
            step=1,
            label="Top-K"
        ),

        gr.Slider(
            minimum=20,
            maximum=150,
            value=50,
            step=10,
            label="Max New Tokens"
        )
    ],

    outputs=gr.Textbox(
        label="Generated Text",
        lines=12
    ),

    title="NovaGPT",

    description=(
        "A character-level GPT-style language model "
        "built from scratch using PyTorch."
    )
)


# -----------------------------
# Launch
# -----------------------------

if __name__ == "__main__":
    demo.launch()