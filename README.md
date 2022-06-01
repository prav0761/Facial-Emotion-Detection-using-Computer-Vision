# Affectnet
Machines have been able to recognize and differentiate between faces for a few years now, but the human face has a purpose outside of identity. Mouths and eyebrows are vital to human communication and allow us to convey tone and emotion without the use of words or gestures. If machines gained the capability to interpret facial expressions into human emotion, it would open a whole new world of sentiment analysis. While the capability is obviously great, this task is hard for many humans to perform. Humans often misinterpret non-verbal signals either due to the similarity of two emotions or due to incorrect assumptions of tone. For our example we are seeking to discriminate between eight key emotions: Happy, Sad, Contempt, Disgust, Fear, Anger, Surprise, and Neutral. As discussed earlier we can see how some of these emotions represent similar feelings, such as Contempt and Disgust, and how other emotions may evoke similar reactions, such as Fear and Surprise. All these factors make this problem difficult to solve, however, our experimentation leads us to believe that it is possible to attain high accuracy in this classification problem.

PROJECT TEAM
UT AUSTIN
1. PRAVEEN RADHAKRISHNAN
2. ANTHONY PHAM
3. AVISH THAKRAL
4. NATHAN PAUL
5. NIKHIL GANESH KRISH
6. JACOB DACH

Background
Our dataset of choice for this problem is AffectNet. This dataset is attractive for image recognition due to its large scale; there are around 1 million images in the dataset, and there is 290k images that have been hand labeled with their depicted emotion. In addition, the AffectNet Dataset contains valence and arousal values which describe emotions in a continuous manner. These values are not commonly found in other facial emotion recognition datasets and allows us to create models that predict an image’s location in this continuous space rather than classify the image within the discrete space described by our eight designated emotions: Neutral, Happy, Sad, Surprise, Anger, Disgust, Fear, and Contempt.
