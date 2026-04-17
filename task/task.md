1. use streamlit help me design a UI platform with the following function:

a) user can upload an image, and you will use open_ai api to recognized the main concept in the image with one word, such as ball, dog, fligh
b) Phoneme segmentation, breaking a word into its individual sounds (phonemes), and visualize it, such as dog → /d/ /ɔ/ /g/
c) generate audio to both the individual sounds (phonemes) and the word
d) voice clone, use this project https://github.com/corentinj/real-time-voice-cloning, to clone the voice into other styles, you can use a few candidate voice, and the user can choose which one to clone

Principles, if the function can be used with open ai api, then use it, 

set your OpenAI API key in the environment before running the app

pip install -r requirements.txt

export OPENAI_API_KEY="your-openai-api-key"

but there are two feature not exists, first Phoneme segmentation, both the word and the sgemeted word need to visualize, second, vice style change, you need to provide few voice style, for voice clone
