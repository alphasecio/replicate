import os, replicate, streamlit as st

# Streamlit app
st.subheader("Replicate Playground")
with st.sidebar:
  replicate_api_token = st.text_input("Replicate API Token", type="password")
  os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
prompt = st.text_input("Prompt")

# If Generate button is clicked
if st.button("Generate"):
  if not replicate_api_token.strip() or not prompt.strip():
    st.error("Please provide the missing fields.")
  else:
    try:
      with st.spinner("Please wait..."):
        # Run stability-ai/stable-diffusion image model on Replicate
        output = replicate.run(
          "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b", 
          input={"prompt": prompt}
          )
        st.image(output)
    except Exception as e:
      st.exception(f"Exception: {e}")
