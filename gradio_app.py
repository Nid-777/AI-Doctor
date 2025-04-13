# if you dont use pipenv uncomment the following:
#from dotenv import load_dotenv
#load_dotenv()

#VoiceBot UI with Gradio
import os
import gradio as gr

from brain import encode_image, analyze_image_with_query
from voice_of_patient import record_audio, transcribe_with_groq
from voice_of_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

#load_dotenv()

system_prompt="""You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_filepath, image_filepath):
    speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                                                 audio_filepath=audio_filepath,
                                                 stt_model="whisper-large-v3")

    # Handle the image input
    if image_filepath:
        doctor_response = analyze_image_with_query(query=system_prompt+speech_to_text_output, encoded_image=encode_image(image_filepath), model="llama-3.2-11b-vision-preview")
    else:
        doctor_response = "No image provided for me to analyze"

    voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath="final.mp3") 

    return speech_to_text_output, doctor_response, voice_of_doctor


# Create the interface
def process_inputs(audio_path, image_path):
    # Simulate processing delay
    transcript = "📝 This is the transcribed audio."
    diagnosis = "💬 Preliminary diagnosis based on image analysis."
    return transcript, diagnosis, "Temp.mp3"

custom_css = """
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    transition: background 0.5s ease;
}
.light-mode {
    background:  #ffffff;
}
.dark-mode {
    background: #1e1e2f;
    color: white;
}

.main-heading {
    text-align: center;
    font-size: 2.6rem ;
    color: #7272c4;
    margin-bottom: 0.21em;

}
.sub-heading {
    text-align: center;
    font-size: 0.95rem;
    text-decoration: underline;
    color:#ccccdb;
    transition: opacity 0.4s ease;
}
.sub-heading:hover {
    opacity: 0.4;
}

textarea, input, .audio-container, .output-box {
    border-radius: 12px;
    border: 1px solid #b2dfdb;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}
.fade-in {
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
"""

with gr.Blocks(css=custom_css, theme="soft") as app:
    # Theme toggle logic
    dark_mode = gr.State(False)

    def toggle_theme(current):
        return not current

    with gr.Column():
                  
        gr.Markdown("<div class='main-heading'>AI Doctor with Vision and Voice</div>")
        gr.Markdown("<div class='sub-heading'>Speak your symptoms & upload image — get early diagnosis❤️ </div>")

        # Toggle Button
        toggle = gr.Button("🌙 Toggle Dark Mode")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Speak Your Symptoms")
                image_input = gr.Image(type="filepath", label="🖼️ Upload Affected Area Image")
                submit_btn = gr.Button("🔍 Diagnose Now")

            with gr.Column():
                output1 = gr.Textbox(label="📝 Transcribed Speech", lines=3, elem_classes="output-box")
                output2 = gr.Textbox(label="💬 Doctor's Diagnosis", lines=4, elem_classes="output-box")
                audio_output = gr.Audio(label="🎧 Doctor's Voice Response")

        status = gr.Textbox(visible=False)

        # On Submit
        def wrapper(audio, img):
            status.update(value="⏳ Processing...", visible=True)
            transcript, diagnosis, voice = process_inputs(audio, img)
            return transcript, diagnosis, voice

        submit_btn.click(fn=wrapper, 
                         inputs=[audio_input, image_input],
                         outputs=[output1, output2, audio_output])

        # Theme toggle JS (for dark/light mode)
        toggle.click(None, dark_mode, dark_mode, 
                     js="""
                     () => {
                         const root = document.querySelector('.gradio-container');
                         root.classList.toggle('dark-mode');
                         root.classList.toggle('light-mode');
                     }
                     """)

app.launch(debug=True)


#http://127.0.0.1:7860