import os, replicate, streamlit as st

# Streamlit app
st.subheader("Replicate Playground")
with st.sidebar:
  replicate_api_token = st.text_input("Replicate API Token", type="password")
  option = st.selectbox("Select Model", [
    "Text: Meta Llama 2 70B",
    "Text: Google Gemma 7B Instruct",
    "Text: Mixtral 8x7B Instruct",
    "Image: Stable Diffusion XL", 
    "Code: Meta Code Llama 70B Instruct", 
    "Music: Meta MusicGen"]
    )

os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
prompt = st.text_input("Prompt", label_visibility="collapsed")

# If Generate button is clicked
if st.button("Generate"):
  if not replicate_api_token.strip() or not prompt.strip():
    st.error("Please provide the missing fields.")
  else:
    try:
      with st.spinner("Please wait..."):
        if option == "Text: Meta Llama 2 70B":
          # Run meta/llama-2-70b-chat model on Replicate
          output = replicate.run(
              "meta/llama-2-70b-chat",
              input={
                  "debug": False,
                  "top_p": 1,
                  "prompt": prompt,
                  "temperature": 0.5,
                  "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                  "max_new_tokens": 500,
                  "min_new_tokens": -1
              },
          )
          st.success(''.join(output))
        elif option == "Text: Google Gemma 7B Instruct":
          # Run google-deepmind/gemma-7b-it model on Replicate
          output = replicate.run(
              "google-deepmind/gemma-7b-it:2790a695e5dcae15506138cc4718d1106d0d475e6dca4b1d43f42414647993d5",
              input={
                  "top_k": 50,
                  "top_p": 0.95,
                  "prompt": prompt,
                  "temperature": 0.7,
                  "max_new_tokens": 512,
                  "min_new_tokens": -1,
                  "repetition_penalty": 1
              },
          )
          st.success(''.join(output))
        elif option == "Text: Mixtral 8x7B Instruct":
          # Run mistralai/mixtral-8x7b-instruct-v0.1 model on Replicate
          output = replicate.run(
              "mistralai/mixtral-8x7b-instruct-v0.1",
              input={
                  "top_k": 50,
                  "top_p": 0.9,
                  "prompt": prompt,
                  "temperature": 0.6,
                  "max_new_tokens": 1024,
                  "prompt_template": "<s>[INST] {prompt} [/INST] ",
                  "presence_penalty": 0,
                  "frequency_penalty": 0
              },
          )
          st.success(''.join(output))
        elif option == "Image: Stable Diffusion XL":
          # Run stability-ai/stable-diffusion image model on Replicate
          output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b", 
            input={"prompt": prompt}
            )
          st.image(output)
        elif option == "Code: Meta Code Llama 70B Instruct":
          # Run meta/codellama-70b-instruct model on Replicate
          output = replicate.run(
              "meta/codellama-70b-instruct:a279116fe47a0f65701a8817188601e2fe8f4b9e04a518789655ea7b995851bf",
              input={
                  "top_k": 10,
                  "top_p": 0.95,
                  "prompt": prompt,
                  "max_tokens": 500,
                  "temperature": 0.8,
                  "system_prompt": "",
                  "repeat_penalty": 1.1,
                  "presence_penalty": 0,
                  "frequency_penalty": 0
              }
          )
          st.success(''.join(output))
        elif option == "Music: Meta MusicGen":
          # Run meta/musicgen model on Replicate
          output = replicate.run(
              "meta/musicgen:b05b1dff1d8c6dc63d14b0cdb42135378dcb87f6373b0d3d341ede46e59e2b38",
              input={
                  "top_k": 250,
                  "top_p": 0,
                  "prompt": prompt,
                  "duration": 33,
                  "temperature": 1,
                  "continuation": False,
                  "model_version": "stereo-large",
                  "output_format": "wav",
                  "continuation_start": 0,
                  "multi_band_diffusion": False,
                  "normalization_strategy": "peak",
                  "classifier_free_guidance": 3
              }
          )
          st.audio(output, format="audio/wav")
    except Exception as e:
      st.exception(f"Exception: {e}")
