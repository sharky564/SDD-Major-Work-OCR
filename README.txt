# SDD-Major-Work-OCR
Major work for 2019 SDD task

OPTICAL CHARACTER RECOGNITION PROGRAM

###############################################################################################################

INSTRUCTIONS FOR SETUP

FOLLOW THE BELOW STEPS PROVIDED YOU HAVEN'T ALREADY DONE THEM BEFORE.

1. DOWNLOAD Python 3.5+ FROM https://www.python.org/downloads/
2. OPEN A TERMINAL (Command Prompt ON Windows, Terminal ON Mac OS OR Linux)
3. TYPE IN 'pip3 install cv2' AND PRESS Enter
4. TYPE IN 'pip3 install numpy' AND PRESS Enter
5. TYPE IN 'pip3 install tensorflow==1.14.0' AND PRESS Enter
6. TYPE IN 'pip3 install editdistance' AND PRESS Enter
7. TYPE IN 'pip3 install argparse' AND PRESS Enter

ONCE ALL THESE LIBRARIES HAE BEEN DOWNLOADED, YOU ARE NOW READY TO RUN THE PROGRAM.

TO DO THIS, OPEN A TERMINAL AND TYPE 'python3 ', THEN CLICK AND DRAG THE Main.py FILE ONTO THE TERMINAL. THIS SHOULD APPEND THE FILE PATH OF THE PROGRAM AFTER THE STRING YOU JUST TYPED OUT. IF THIS DIDN'T WORK, YOU CAN ALSO TYPE IT MANUALLY. ONCE THE FILE PATH IS ADDED, PRESS Space AND TYPE --help OR -h TO DISPLAY THE OTHER INTERNAL HELP COMMANDS.

###############################################################################################################

THE COMMANDS THAT ARE ALLOWED TO PARSE ARE:

1. --help
THIS DISPLAYS WHAT ALL THE COMMANDS ARE AS WELL AS WHAT THEY DO.

2. --train
THIS TRAINS THE NEURAL NETWORK FROM THE DATASET THAT IT HAS ALREADY BEEN PROVIDED in /data/words/. THIS DATASET IS THE IAM HANDWRITING DATASET, AND SHOULD NOT BE USED FOR ANY PURPOSES APART FROM FOR TRAINING THIS DATASET. THE TRAIN FUNCTION EITHER PULLS A SAVED MODEL FROM /save/ OR IT STARTS FROM SCRATCH. THIS PROCESS USUALLY TAKES OVER 24 HOURS TO TRAIN FULLY ON A COMPUTER WITH 4GB RAM, AND WILL STOP TRAINING WHEN THE SYSTEM DEEMS ITSELF WELL TRAINED, SO JUST LEAVE IT ON RUNNING UNTIL IT STOPS ON ITS OWN REGARD. 

3. --test
THIS TESTS THE NEURAL NETWORK WITH IMAGES FROM THE PROVIDED DATASET IN /data/words/. THIS TAKES ABOUT 4 MINUTES. IT WILL THEN OUTPUT ITS CHARACTER ERROR RATE AND WORD ACCURACY ON THE TERMINAL, AND ALSO OUTPUT IT ON A TEXT FILE IN /save/accuracy.txt. 

4. --use
THIS ALLOWS THE USER TO INPUT THEIR OWN HANDWRITING IMAGE INTO THE NEURAL NETWORK AND GET A TRANSLATION FOR IT. TO USE THIS, THE USER MUST REPLACE THE PROVIDED FILE /data/test.png WITH THEIR OWN IMAGE, WHICH SHOULD BE RENAMED test.png, THEN TYPE IN --use ON THE TERMINAL. THE NETWORK WILL OUTPUT THE TRANSLATION ON THE TERMINAL, AND ALSO WRITE A CSV FILE IN /output/ WITH THE TENSOR IT HAD TO DECODE (THIS IS FOR PEOPLE WHO WANT TO ANALYSE THE SYSTEM IN MORE DETAIL).

###############################################################################################################

NOTE THAT RUNNING THESE COMMANDS INITIALLY WHILL TAKE A WHILE DUE TO THE SIZE AND COMPLEXITY OF THESE FILES, AND A LOT OF DEPRECATION WARNINGS WILL POP UP AS TENSORFLOW IS UNDERGOING A LOT OF CHANGES SO THESE COMMANDS ARE GOING TO BE DEPRECATED IN FUTURE. HOWEVER, THESE WILL NOT CHANGE THE OUTPUT OF THE NETWORK, SO YOU CAN IGNORE THESE.

DO NOT ATTEMPT TO MODIFY THIS CODE UNLESS YOU KNOW WHAT YOU'RE DOING AS IT IS VERY TRICKY TO WORK WITH.

IF YOU WANT TO RE-TRAIN THE NETWORK FROM SCRATCH, DELETE ALL THE FILES IN /save/, THEN USE THE --train COMMAND IN TERMINAL.

###############################################################################################################

WHEN RUNNING THESE COMMANDS, THESE CAN BE WHAT YOUR EXPECTED OUTPUTS LOOK LIKE ON THE TERMINAL:

1. --help

usage: Main.py [-h] [--train] [--test] [--use]

optional arguments:
  -h, --help  show this help message and exit
  --train     Train the Neural Network on the given dataset. For more
              information, read the README file.
  --test      Test the Neural Network on the given dataset. For more
              information, read the README file.
  --use       Uses and saves the outputs of the Neural Networks onto CSV
              files. For more information, read the README file.

2. --train

Training the Neural Network. Beginning Epoch 1
Batch 1 / 500 trained. Current cost: 13.845441
Batch 2 / 500 trained. Current cost: 25.531614
Batch 3 / 500 trained. Current cost: 74.298836
Batch 4 / 500 trained. Current cost: 72.0811
Batch 5 / 500 trained. Current cost: 39.162804
Batch 6 / 500 trained. Current cost: 69.237656
...
Batch 497 / 500 trained. Current cost: 14.272595
Batch 498 / 500 trained. Current cost: 14.713428
Batch 499 / 500 trained. Current cost: 12.422189
Batch 500 / 500 trained. Current cost: 16.338842
Testing the Neural Network.
Batch 1 / 115 tested.
Batch 2 / 115 tested.
Batch 3 / 115 tested.
Batch 4 / 115 tested.
Batch 5 / 115 tested.
Batch 6 / 115 tested.
...
Batch 114 / 115 tested.
Batch 115 / 115 tested.
Character error rate: 92.716000% | Word accuracy: 0.347826%

3. --test

Testing the Neural Network.
Batch 1 / 115 tested.
Batch 2 / 115 tested.
Batch 3 / 115 tested.
Batch 4 / 115 tested.
Batch 5 / 115 tested.
Batch 6 / 115 tested.
...
Batch 114 / 115 tested.
Batch 115 / 115 tested.
Character error rate: 92.716000% | Word accuracy: 0.347826%

4. --use

Translation: ['a']
