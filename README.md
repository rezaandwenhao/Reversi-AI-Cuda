# Reversi-AI-Cuda

## Acknowledgement
The game logic is based on https://github.com/rodolfoams/reversi-c. We rewrote the code and implemented the AI players and the random player on top of its terminal user inputting steps.

## How to run the program
One can directly open the reversiAI.ipynb in the Google Colab. Select runtime as GPU and run all cells. 

Essentially, the reversi.cu is compiled by 
```
nvcc -arch=sm_35 -rdc=true reversiAI.cu -o a
```

and run by 
```
./a <AI_Player Level> <AI play with human or not>
```

The program takes 2 arguments: <AI_Player Level> and \<AI play with human or not\> \

The first argument defines the level of AI player to play against: \
<AI_Player Level>: 0-Easy, 1-Medium, 2-Easy_CUDA, 3-Medium_CUDA

The second argument defines whether user wants to play with AI or let random player to play against AI: \
\<AI play with human or not\>: 0-Human plays with AI, 1-Random player plays with AI.
