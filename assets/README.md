# Prompt Example
## Speech
### Text-To-Speech
Input Example : Generate a speech with text "here we go"<br />
Output:<br />
![](tts.png)<br />
Audio:<br />
<audio src="fd5cf55e.wav" controls></audio><br />

### Style Transfer Text-To-Speech
First upload your audio(.wav)<br />
Input Example : Speak using the voice of this audio. The text is "here we go".<br />
Output:<br />
![](style_transfer_tts.png)<br />

### Speech Recognition
First upload your audio(.wav)<br />
Audio Example :<br />
<audio src="Track 4.wav" controls></audio><br />
Input Example : Generate the text of this speech<br />
Output:<br />
![](asr.png)<br />

## Sing
### Text-To-Sing
Input example : please generate a piece of singing voice. Text sequence is 小酒窝长睫毛AP是你最美的记号. Note sequence is C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4. Note duration sequence is 0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340.<br />
Output:<br />
![](t2s.png)<br />
Audio:<br />
<audio src="2bf90e35.wav" controls></audio><br />

## Audio
### Text-To-Audio
Input Example : Generate an audio of a piano playing<br />
Output:<br />
![](t2a.png)<br />
Audio:<br />
<audio src="b973e878.wav" controls></audio><br />

### Audio Inpainting
First upload your audio(.wav)<br />
Audio Example :<br />
<audio src="drums-and-music-playing-with-a-man-speaking.wav" controls></audio><br />
Input Example : I want to inpaint this audio.<br />
Output:<br />
![](inpaint-1.png)<br />
Then you can press the "Predict Masked Place" button<br />
Output:<br />
![](inpaint-2.png)<br />
Output Audio:<br />
<audio src="7cb0d24f.wav" controls></audio><br />

### Image-To-Audio
First upload your image(.png)<br />
Input Example : Generate the audio of this image<br />
Output:<br />
![](i2a-2.png)<br />
Audio:<br />
<audio src="5d67d1b9.wav" controls></audio><br />

### Audio-To-Text
First upload your audio(.wav)<br />
Audio Example :<br />
<audio src="a-group-of-sheep-are-baaing.wav" controls></audio><br />
Input Example : Please tell me the text description of this audio.<br />
Output:<br />
![](a2i.png)<br />


### Sound Detection
First upload your audio(.wav)<br />
Audio Example :<br />
<audio src="mix.wav" controls></audio><br />
Input Example : What events does this audio include?<br />
Output:<br />
![](detection.png)<br />

### Mono audio to Binaural Audio
First upload your audio(.wav)<br />
<audio src="mix.wav" controls></audio><br />
Input Example: Transfer the mono speech to a binaural one audio.<br />
Output:<br />
![](m2b.png)<br />

### Target Sound Detection
Fisrt upload your audio(.wav)<br />
<audio src="mix.wav" controls></audio><br />
Input Example: please help me detect the target sound in the audio based on desription: “I want to detect Applause event”<br />
Output:<br />
![](tsd.png)<br />

### Sound Extraction
First upload your audio(.wav)<br />
<audio src="mix.wav" controls></audio><br />
Input Example: Please help me extract the sound events from the audio based on the description: "a person shouts nearby and then emergency vehicle sirens sounds"<br />
Output:<br />
![](sound_extraction.png)<br />
