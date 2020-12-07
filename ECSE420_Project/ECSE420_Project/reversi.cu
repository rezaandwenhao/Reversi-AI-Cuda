#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WHITE    0
#define BLACK    1
#define EMPTY    2
#define PLAYABLE 3

#define WHITE_MARKER    "  X  "
#define BLACK_MARKER    "  O  "
#define EMPTY_MARKER    "     "
#define PLAYABLE_MARKER "  .  "

#define FALSE 0
#define TRUE  1

#define AI_PLAYER 0
#define STEPS_THINK_AHEAD 3  // There are some bugs that the AI does not work when think more than 3 steps ahead
enum AI_LEVEL {Easy, Medium, Easy_CUDA, Medium_CUDA};
int ai;
int humanPlay;

const char* row_names = "01234567";
const char* col_names = "01234567";

char board[8][8];
int playable_direction[8][8][8];
int current_player;
int game_ended = FALSE;
int skipped_turn = FALSE;
int wrong_move = FALSE;
int has_valid_move = FALSE;
int scores[2];
double total_move_time = 0.0;

__device__ int is_valid_position(int i, int j)
{
	if (i < 0 || i >= 8 || j < 0 || j >= 8) return FALSE;
	return TRUE;
}

__device__ int distance(int i1, int j1, int i2, int j2)
{
	int di = abs(i1 - i2), dj = abs(j1 - j2);
	if (di > 0) return di;
	return dj;
}

// Update the copied board for copy_playable_direction
__device__ int is_playable_simulate(int i, int j, char* copy_board, int* copy_playable_direction, int tmp_current_player)
{
	// memset( copy_playable_direction[i][j], 0, 8 );
	for (int m = 0; m < 8; ++m)
	{
		copy_playable_direction[i * 64 + j * 8 + m] = 0;
	}
	if (!is_valid_position(i, j)) return FALSE;
	if (copy_board[i * 8 + j] != EMPTY) return FALSE;
	int playable = FALSE;

	int opposing_player = (tmp_current_player + 1) % 2;

	// Test UL diagonal
	int i_it = i - 1, j_it = j - 1;
	while (is_valid_position(i_it, j_it) && copy_board[i_it * 8 + j_it] == opposing_player)
	{
		i_it -= 1;
		j_it -= 1;
	}
	if (is_valid_position(i_it, j_it) && distance(i, j, i_it, j_it) > 1 && copy_board[i_it * 8 + j_it] == tmp_current_player)
	{
		copy_playable_direction[i * 64 + j * 8 + 0] = 1;
		playable = TRUE;
	}

	// Test UP path
	i_it = i - 1, j_it = j;
	while (is_valid_position(i_it, j_it) && copy_board[i_it * 8 + j_it] == opposing_player)
		i_it -= 1;

	if (is_valid_position(i_it, j_it) && distance(i, j, i_it, j_it) > 1 && copy_board[i_it * 8 + j_it] == tmp_current_player)
	{
		copy_playable_direction[i * 64 + j * 8 + 1] = 1;
		playable = TRUE;
	}

	// Test UR diagonal
	i_it = i - 1, j_it = j + 1;
	while (is_valid_position(i_it, j_it) && copy_board[i_it * 8 + j_it] == opposing_player)
	{
		i_it -= 1;
		j_it += 1;
	}
	if (is_valid_position(i_it, j_it) && distance(i, j, i_it, j_it) > 1 && copy_board[i_it * 8 + j_it] == tmp_current_player)
	{
		copy_playable_direction[i * 64 + j * 8 + 2] = 1;
		playable = TRUE;
	}

	// Test LEFT path
	i_it = i, j_it = j - 1;
	while (is_valid_position(i_it, j_it) && copy_board[i_it * 8 + j_it] == opposing_player)
		j_it -= 1;

	if (is_valid_position(i_it, j_it) && distance(i, j, i_it, j_it) > 1 && copy_board[i_it * 8 + j_it] == tmp_current_player)
	{
		copy_playable_direction[i * 64 + j * 8 + 3] = 1;
		playable = TRUE;
	}

	// Test RIGHT path
	i_it = i, j_it = j + 1;
	while (is_valid_position(i_it, j_it) && copy_board[i_it * 8 + j_it] == opposing_player)
		j_it += 1;

	if (is_valid_position(i_it, j_it) && distance(i, j, i_it, j_it) > 1 && copy_board[i_it * 8 + j_it] == tmp_current_player)
	{
		copy_playable_direction[i * 64 + j * 8 + 4] = 1;
		playable = TRUE;
	}

	// Test DL diagonal
	i_it = i + 1, j_it = j - 1;
	while (is_valid_position(i_it, j_it) && copy_board[i_it * 8 + j_it] == opposing_player)
	{
		i_it += 1;
		j_it -= 1;
	}
	if (is_valid_position(i_it, j_it) && distance(i, j, i_it, j_it) > 1 && copy_board[i_it * 8 + j_it] == tmp_current_player)
	{
		copy_playable_direction[i * 64 + j * 8 + 5] = 1;
		playable = TRUE;
	}

	// Test DOWN path
	i_it = i + 1, j_it = j;
	while (is_valid_position(i_it, j_it) && copy_board[i_it * 8 + j_it] == opposing_player)
		i_it += 1;

	if (is_valid_position(i_it, j_it) && distance(i, j, i_it, j_it) > 1 && copy_board[i_it * 8 + j_it] == tmp_current_player)
	{
		copy_playable_direction[i * 64 + j * 8 + 6] = 1;
		playable = TRUE;
	}

	// Test DR diagonal
	i_it = i + 1, j_it = j + 1;
	while (is_valid_position(i_it, j_it) && copy_board[i_it * 8 + j_it] == opposing_player)
	{
		i_it += 1;
		j_it += 1;
	}
	if (is_valid_position(i_it, j_it) && distance(i, j, i_it, j_it) > 1 && copy_board[i_it * 8 + j_it] == tmp_current_player)
	{
		copy_playable_direction[i * 64 + j * 8 + 7] = 1;
		playable = TRUE;
	}
	return playable;
}

__device__ int capture_potential_pieces(int i, int j, char* copy_board, int curr_potential_player, int* copy_playable_direction)
{
	int opposing_player = (curr_potential_player + 1) % 2;
	int potential_score = 0;
	int i_it, j_it;

	// Capture UL diagonal
	if (copy_playable_direction[i * 64 + j * 8 + 0])
	{
		i_it = i - 1, j_it = j - 1;
		while (copy_board[i_it * 8 + j_it] == opposing_player)
		{
			copy_board[i_it * 8 + j_it] = curr_potential_player;
			potential_score++;
			i_it -= 1;
			j_it -= 1;
		}
	}

	// Capture UP path
	if (copy_playable_direction[i * 64 + j * 8 + 1])
	{
		i_it = i - 1, j_it = j;
		while (copy_board[i_it * 8 + j_it] == opposing_player)
		{
			copy_board[i_it * 8 + j_it] = curr_potential_player;
			potential_score++;
			i_it -= 1;
		}
	}

	// Capture UR diagonal
	if (copy_playable_direction[i * 64 + j * 8 + 2])
	{
		i_it = i - 1, j_it = j + 1;
		while (copy_board[i_it * 8 + j_it] == opposing_player)
		{
			copy_board[i_it * 8 + j_it] = curr_potential_player;
			potential_score++;
			i_it -= 1;
			j_it += 1;
		}
	}

	// Capture LEFT path
	if (copy_playable_direction[i * 64 + j * 8 + 3])
	{
		i_it = i, j_it = j - 1;
		while (copy_board[i_it * 8 + j_it] == opposing_player)
		{
			copy_board[i_it * 8 + j_it] = curr_potential_player;
			potential_score++;
			j_it -= 1;
		}
	}

	// Capture RIGHT path
	if (copy_playable_direction[i * 64 + j * 8 + 4])
	{
		i_it = i, j_it = j + 1;
		while (copy_board[i_it * 8 + j_it] == opposing_player)
		{
			copy_board[i_it * 8 + j_it] = curr_potential_player;
			potential_score++;
			j_it += 1;
		}
	}

	// Capture DL diagonal
	if (copy_playable_direction[i * 64 + j * 8 + 5])
	{
		i_it = i + 1, j_it = j - 1;
		while (copy_board[i_it * 8 + j_it] == opposing_player)
		{
			copy_board[i_it * 8 + j_it] = curr_potential_player;
			potential_score++;
			i_it += 1;
			j_it -= 1;
		}
	}

	// Capture DOWN path
	if (copy_playable_direction[i * 64 + j * 8 + 6])
	{
		i_it = i + 1, j_it = j;
		while (copy_board[i_it * 8 + j_it] == opposing_player)
		{
			copy_board[i_it * 8 + j_it] = curr_potential_player;
			potential_score++;
			i_it += 1;
		}
	}

	// Capture DR diagonal
	if (copy_playable_direction[i * 64 + j * 8 + 7])
	{
		i_it = i + 1, j_it = j + 1;
		while (copy_board[i_it * 8 + j_it] == opposing_player)
		{
			copy_board[i_it * 8 + j_it] = curr_potential_player;
			potential_score++;
			i_it += 1;
			j_it += 1;
		}
	}
	return potential_score;
}

// Mark playable positions for the simulated copied board.
__device__ void mark_playable_positions_simulate(char* copy_board, int* copy_playable_direction, int tmp_current_player)
{
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
      int idx = i * 8 + j;
			if (copy_board[idx] == PLAYABLE)
				copy_board[idx] = EMPTY;
			if (is_playable_simulate(i, j, copy_board, copy_playable_direction, tmp_current_player))
			{
				copy_board[idx] = PLAYABLE;
			}
		}
	}
}

__global__ void predict_next_move(char* cboard, int* cplayable_direction, int current_player, int count, int* maxScore) {
  if (count == 0) {
      return;
  }
  int idx = threadIdx.x;
	if (idx < 64)
	{
		int i = idx / 8;
		int j = idx % 8;
		//printf("board, %d\n", board[i*8 + j]);
		if (cboard[i * 8 + j] == 3) {
      char* copy_board2;
      int* copy_playable_direction2;
      cudaMalloc((char**)&copy_board2, 64 * sizeof(char));
      cudaMalloc((int**)&copy_playable_direction2, 512 * sizeof(int));
			memcpy(copy_board2, cboard, sizeof(char) * 8 * 8);
			memcpy(copy_playable_direction2, cplayable_direction, sizeof(int) * 8 * 8 * 8);
      copy_board2[i * 8 + j] = current_player;
			int s = capture_potential_pieces(i, j, copy_board2, current_player, copy_playable_direction2);
      mark_playable_positions_simulate(copy_board2, copy_playable_direction2, (current_player + 1) % 2);
      int* res2;
      cudaMalloc((int**)&res2, 64*sizeof(int));
      for (int k = 0; k < 64; k++) {
        res2[k] = -100*count;
      }
      predict_next_move<<<1, 64>>>(copy_board2, copy_playable_direction2, (current_player + 1) % 2, count-1, res2);
      cudaDeviceSynchronize();

      int max = -100*count;
      for (int k = 0; k < 64; k++) {
        if (res2[k] >= max) {
          max = res2[k];
        }   
      }
      if (max == -100*count) max = 0;
      maxScore[idx] = s - max;
      
      cudaFree(copy_board2);
      cudaFree(copy_playable_direction2);
      cudaFree(res2);
		}
	}
}


typedef struct
{
	int i = 0;
	int j = 0;
	int max = -200;
} move;

// Think ahead 3 steps
__global__ void get_mediumAI_move(char* cboard, int* cplayable_direction, int current_player, move* moves)
{
	int idx = threadIdx.x;
	if (idx < 64)
	{
		int i = idx / 8;
		int j = idx % 8;
		//printf("board, %d\n", board[i*8 + j]);
		if (cboard[i * 8 + j] == 3) {
      char* copy_board2;
      int* copy_playable_direction2;
      cudaMalloc((char**)&copy_board2, 64 * sizeof(char));
      cudaMalloc((int**)&copy_playable_direction2, 512 * sizeof(int));
			memcpy(copy_board2, cboard, sizeof(char) * 8 * 8);
			memcpy(copy_playable_direction2, cplayable_direction, sizeof(int) * 8 * 8 * 8);
      copy_board2[i * 8 + j] = current_player;
			int s = capture_potential_pieces(i, j, copy_board2, current_player, copy_playable_direction2);
      mark_playable_positions_simulate(copy_board2, copy_playable_direction2, (current_player + 1) % 2);
      int* res2;
      cudaMalloc((int**)&res2, 64*sizeof(int));
      for (int k = 0; k < 64; k++) {
        res2[k] = -1000;
      }
      predict_next_move<<<1, 64>>>(copy_board2, copy_playable_direction2, (current_player + 1) % 2, STEPS_THINK_AHEAD-1, res2);
      cudaDeviceSynchronize();

      int max = -1000;
      for (int k = 0; k < 64; k++) {
        if (res2[k] >= max) {
          max = res2[k];
        }   
      }
      if (max == -1000) max = 0;
			moves[idx].max = s - max;
      //printf("res2: %d\n", s-max);
			moves[idx].i = i;
      //printf("i: %d\n", i);
			moves[idx].j = j;
      //printf("j: %d\n", j);

      cudaFree(copy_board2);
      cudaFree(copy_playable_direction2);
      cudaFree(res2);
		}
	}
}

__global__ void get_easyAI_move(char* board, int* playable_direction, int current_player, move* moves)
{
	int idx = threadIdx.x;
	if (idx < 64)
	{
		int i = idx / 8;
		int j = idx % 8;
		//printf("board, %d\n", board[i*8 + j]);
		if (board[i * 8 + j] == 3) {
			char copy_board[64];
			int copy_playable_direction[512];
			memcpy(copy_board, board, sizeof(char) * 8 * 8);
			memcpy(copy_playable_direction, playable_direction, sizeof(int) * 8 * 8 * 8);
			int s = capture_potential_pieces(i, j, copy_board, current_player, copy_playable_direction);

			//printf("easy score: %d here, row: %d, column: %d\n", s, i, j);
			moves[idx].max = s;
			moves[idx].i = i;
			moves[idx].j = j;
		}
	}
}


////////////////////////////////////////////////////////////////////////////// Seq below ////////////////////////////////////////////////////////////////////////////////////////////

void init_game()
{
	memset(board, EMPTY, sizeof(board));
	board[3][3] = BLACK;
	board[4][4] = BLACK;
	board[3][4] = WHITE;
	board[4][3] = WHITE;
	scores[WHITE] = 2;
	scores[BLACK] = 2;
	game_ended = FALSE;
	skipped_turn = FALSE;
	wrong_move = FALSE;
	has_valid_move = FALSE;
	current_player = BLACK;
}

int is_valid_position_cpu(int i, int j)
{
	if (i < 0 || i >= 8 || j < 0 || j >= 8) return FALSE;
	return TRUE;
}

int distance_cpu(int i1, int j1, int i2, int j2)
{
	int di = abs(i1 - i2), dj = abs(j1 - j2);
	if (di > 0) return di;
	return dj;
}

int is_playable_cpu(int i, int j)
{
	// memset( playable_direction[i][j], 0, 8 );
	for (int m = 0; m < 8; ++m)
	{
		playable_direction[i][j][m] = 0;
	}
	if (!is_valid_position_cpu(i, j)) return FALSE;
	if (board[i][j] != EMPTY) return FALSE;
	int playable = FALSE;

	int opposing_player = (current_player + 1) % 2;

	// Test UL diagonal
	int i_it = i - 1, j_it = j - 1;
	while (is_valid_position_cpu(i_it, j_it) && board[i_it][j_it] == opposing_player)
	{
		i_it -= 1;
		j_it -= 1;
	}
	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && board[i_it][j_it] == current_player)
	{
		playable_direction[i][j][0] = 1;
		playable = TRUE;
	}

	// Test UP path
	i_it = i - 1, j_it = j;
	while (is_valid_position_cpu(i_it, j_it) && board[i_it][j_it] == opposing_player)
		i_it -= 1;

	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && board[i_it][j_it] == current_player)
	{
		playable_direction[i][j][1] = 1;
		playable = TRUE;
	}

	// Test UR diagonal
	i_it = i - 1, j_it = j + 1;
	while (is_valid_position_cpu(i_it, j_it) && board[i_it][j_it] == opposing_player)
	{
		i_it -= 1;
		j_it += 1;
	}
	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && board[i_it][j_it] == current_player)
	{
		playable_direction[i][j][2] = 1;
		playable = TRUE;
	}

	// Test LEFT path
	i_it = i, j_it = j - 1;
	while (is_valid_position_cpu(i_it, j_it) && board[i_it][j_it] == opposing_player)
		j_it -= 1;

	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && board[i_it][j_it] == current_player)
	{
		playable_direction[i][j][3] = 1;
		playable = TRUE;
	}

	// Test RIGHT path
	i_it = i, j_it = j + 1;
	while (is_valid_position_cpu(i_it, j_it) && board[i_it][j_it] == opposing_player)
		j_it += 1;

	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && board[i_it][j_it] == current_player)
	{
		playable_direction[i][j][4] = 1;
		playable = TRUE;
	}

	// Test DL diagonal
	i_it = i + 1, j_it = j - 1;
	while (is_valid_position_cpu(i_it, j_it) && board[i_it][j_it] == opposing_player)
	{
		i_it += 1;
		j_it -= 1;
	}
	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && board[i_it][j_it] == current_player)
	{
		//printf("%d, %d, %d, %d\n", i, j, i_it, j_it);
		playable_direction[i][j][5] = 1;
		playable = TRUE;
	}

	// Test DOWN path
	i_it = i + 1, j_it = j;
	while (is_valid_position_cpu(i_it, j_it) && board[i_it][j_it] == opposing_player)
		i_it += 1;

	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && board[i_it][j_it] == current_player)
	{
		playable_direction[i][j][6] = 1;
		playable = TRUE;
	}

	// Test DR diagonal
	i_it = i + 1, j_it = j + 1;
	while (is_valid_position_cpu(i_it, j_it) && board[i_it][j_it] == opposing_player)
	{
		i_it += 1;
		j_it += 1;
	}
	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && board[i_it][j_it] == current_player)
	{
		playable_direction[i][j][7] = 1;
		playable = TRUE;
	}
	return playable;
}

int is_playable_simulate_cpu(int i, int j, char copy_board[8][8], int copy_playable_direction[8][8][8], int tmp_current_player)
{
	// memset( copy_playable_direction[i][j], 0, 8 );
	for (int m = 0; m < 8; ++m)
	{
		copy_playable_direction[i][j][m] = 0;
	}
	if (!is_valid_position_cpu(i, j)) return FALSE;
	if (copy_board[i][j] != EMPTY) return FALSE;
	int playable = FALSE;

	int opposing_player = (tmp_current_player + 1) % 2;

	// Test UL diagonal
	int i_it = i - 1, j_it = j - 1;
	while (is_valid_position_cpu(i_it, j_it) && copy_board[i_it][j_it] == opposing_player)
	{
		i_it -= 1;
		j_it -= 1;
	}
	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && copy_board[i_it][j_it] == tmp_current_player)
	{
		copy_playable_direction[i][j][0] = 1;
		playable = TRUE;
	}

	// Test UP path
	i_it = i - 1, j_it = j;
	while (is_valid_position_cpu(i_it, j_it) && copy_board[i_it][j_it] == opposing_player)
		i_it -= 1;

	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && copy_board[i_it][j_it] == tmp_current_player)
	{
		copy_playable_direction[i][j][1] = 1;
		playable = TRUE;
	}

	// Test UR diagonal
	i_it = i - 1, j_it = j + 1;
	while (is_valid_position_cpu(i_it, j_it) && copy_board[i_it][j_it] == opposing_player)
	{
		i_it -= 1;
		j_it += 1;
	}
	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && copy_board[i_it][j_it] == tmp_current_player)
	{
		copy_playable_direction[i][j][2] = 1;
		playable = TRUE;
	}

	// Test LEFT path
	i_it = i, j_it = j - 1;
	while (is_valid_position_cpu(i_it, j_it) && copy_board[i_it][j_it] == opposing_player)
		j_it -= 1;

	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && copy_board[i_it][j_it] == tmp_current_player)
	{
		copy_playable_direction[i][j][3] = 1;
		playable = TRUE;
	}

	// Test RIGHT path
	i_it = i, j_it = j + 1;
	while (is_valid_position_cpu(i_it, j_it) && copy_board[i_it][j_it] == opposing_player)
		j_it += 1;

	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && copy_board[i_it][j_it] == tmp_current_player)
	{
		copy_playable_direction[i][j][4] = 1;
		playable = TRUE;
	}

	// Test DL diagonal
	i_it = i + 1, j_it = j - 1;
	while (is_valid_position_cpu(i_it, j_it) && copy_board[i_it][j_it] == opposing_player)
	{
		i_it += 1;
		j_it -= 1;
	}
	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && copy_board[i_it][j_it] == tmp_current_player)
	{
		copy_playable_direction[i][j][5] = 1;
		playable = TRUE;
	}

	// Test DOWN path
	i_it = i + 1, j_it = j;
	while (is_valid_position_cpu(i_it, j_it) && copy_board[i_it][j_it] == opposing_player)
		i_it += 1;

	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && copy_board[i_it][j_it] == tmp_current_player)
	{
		copy_playable_direction[i][j][6] = 1;
		playable = TRUE;
	}

	// Test DR diagonal
	i_it = i + 1, j_it = j + 1;
	while (is_valid_position_cpu(i_it, j_it) && copy_board[i_it][j_it] == opposing_player)
	{
		i_it += 1;
		j_it += 1;
	}
	if (is_valid_position_cpu(i_it, j_it) && distance_cpu(i, j, i_it, j_it) > 1 && copy_board[i_it][j_it] == tmp_current_player)
	{
		copy_playable_direction[i][j][7] = 1;
		playable = TRUE;
	}
	return playable;
}

void mark_playable_positions_cpu()
{
	has_valid_move = FALSE;
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			if (board[i][j] == PLAYABLE)
				board[i][j] = EMPTY;
			if (is_playable_cpu(i, j))
			{
				board[i][j] = PLAYABLE;
				has_valid_move = TRUE;
			}
		}
	}
}

void mark_playable_positions_simulate_cpu(char copy_board[8][8], int copy_playable_direction[8][8][8], int tmp_current_player)
{
	has_valid_move = FALSE;
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			if (copy_board[i][j] == PLAYABLE)
				copy_board[i][j] = EMPTY;
			if (is_playable_simulate_cpu(i, j, copy_board, copy_playable_direction, tmp_current_player))
			{
				copy_board[i][j] = PLAYABLE;
				has_valid_move = TRUE;
			}
		}
	}
}

void draw_board_cpu()
{
	printf("     %c     %c     %c     %c     %c     %c     %c     %c\n", col_names[0], col_names[1], col_names[2], col_names[3], col_names[4], col_names[5], col_names[6], col_names[7]);
	printf("   _____ _____ _____ _____ _____ _____ _____ _____\n");
	for (int i = 0; i < 8; ++i)
	{
		printf("  |     |     |     |     |     |     |     |     |\n");
		printf("%c |", row_names[i]);
		for (int j = 0; j < 8; ++j)
		{
			if (board[i][j] == WHITE)
			{
				printf("%s", WHITE_MARKER);
			}
			else if (board[i][j] == BLACK)
			{
				printf("%s", BLACK_MARKER);
			}
			else if (board[i][j] == PLAYABLE)
			{
				printf("%s", PLAYABLE_MARKER);
			}
			else
			{
				printf("%s", EMPTY_MARKER);
			}
			printf("|");
		}
		printf("\n");
		printf("  |_____|_____|_____|_____|_____|_____|_____|_____|\n");
	}
	printf("\n");
}

void display_wrong_move_cpu()
{
	if (wrong_move)
	{
		printf("You entered an invalid move!\n");
		wrong_move = FALSE;
	}
}

void display_current_player_cpu()
{
	printf("Current player:");
	if (current_player == WHITE)
		printf("%s", WHITE_MARKER);
	else
		printf("%s", BLACK_MARKER);
	printf("\n");
}

void change_current_player_cpu()
{
	current_player = (current_player + 1) % 2;
}

void prompt_move_cpu(int* p_row, int* p_column)
{
	printf("Enter row [0-7] and column [0-7] separated by a single space (eg.: 2 4).\n");
	scanf("%d %d", p_row, p_column);
}

void capture_pieces_cpu(int i, int j)
{
	int opposing_player = (current_player + 1) % 2;
	int i_it, j_it;
	// Capture UL diagonal
	if (playable_direction[i][j][0])
	{
		i_it = i - 1, j_it = j - 1;
		while (board[i_it][j_it] == opposing_player)
		{
			board[i_it][j_it] = current_player;
			scores[current_player]++;
			scores[opposing_player]--;
			i_it -= 1;
			j_it -= 1;
			if (scores[opposing_player] < 0) printf("problem 1\n");
		}
	}

	// Capture UP path
	if (playable_direction[i][j][1])
	{
		i_it = i - 1, j_it = j;
		while (board[i_it][j_it] == opposing_player)
		{
			board[i_it][j_it] = current_player;
			scores[current_player]++;
			scores[opposing_player]--;
			i_it -= 1;
			if (scores[opposing_player] < 0) printf("problem 2\n");
		}
	}

	// Capture UR diagonal
	if (playable_direction[i][j][2])
	{
		i_it = i - 1, j_it = j + 1;
		while (board[i_it][j_it] == opposing_player)
		{
			board[i_it][j_it] = current_player;
			scores[current_player]++;
			scores[opposing_player]--;
			i_it -= 1;
			j_it += 1;
			if (scores[opposing_player] < 0) printf("problem 3\n");
		}
	}

	// Capture LEFT path
	if (playable_direction[i][j][3])
	{
		i_it = i, j_it = j - 1;
		while (board[i_it][j_it] == opposing_player)
		{
			board[i_it][j_it] = current_player;
			scores[current_player]++;
			scores[opposing_player]--;
			j_it -= 1;
			if (scores[opposing_player] < 0) printf("problem 4\n");
		}
	}

	// Capture RIGHT path
	if (playable_direction[i][j][4])
	{
		i_it = i, j_it = j + 1;
		while (board[i_it][j_it] == opposing_player)
		{
			board[i_it][j_it] = current_player;
			scores[current_player]++;
			scores[opposing_player]--;
			j_it += 1;
			if (scores[opposing_player] < 0) printf("problem 5\n");
		}
	}

	// Capture DL diagonal
	if (playable_direction[i][j][5])
	{
		i_it = i + 1, j_it = j - 1;
		while (board[i_it][j_it] == opposing_player)
		{
			board[i_it][j_it] = current_player;
			scores[current_player]++;
			scores[opposing_player]--;
			i_it += 1;
			j_it -= 1;
			if (scores[opposing_player] < 0) printf("problem 6\n");
		}
	}

	// Capture DOWN path
	if (playable_direction[i][j][6])
	{
		i_it = i + 1, j_it = j;
		while (board[i_it][j_it] == opposing_player)
		{
			board[i_it][j_it] = current_player;
			scores[current_player]++;
			scores[opposing_player]--;
			i_it += 1;
			if (scores[opposing_player] < 0) printf("problem 7\n");
		}
	}

	// Capture DR diagonal
	if (playable_direction[i][j][7])
	{
		i_it = i + 1, j_it = j + 1;
		while (board[i_it][j_it] == opposing_player)
		{
			board[i_it][j_it] = current_player;
			scores[current_player]++;
			scores[opposing_player]--;
			i_it += 1;
			j_it += 1;
			if (scores[opposing_player] < 0) printf("problem 8\n");
		}
	}
}

int capture_potential_pieces_cpu(int i, int j, char copy_board[8][8], int curr_potential_player, int copy_playable_direction[8][8][8])
{
	int opposing_player = (curr_potential_player + 1) % 2;
	int potential_score = 0;
	int i_it, j_it;

	// Capture UL diagonal
	if (copy_playable_direction[i][j][0])
	{
		i_it = i - 1, j_it = j - 1;
		while (copy_board[i_it][j_it] == opposing_player)
		{
			copy_board[i_it][j_it] = curr_potential_player;
			potential_score++;
			i_it -= 1;
			j_it -= 1;
		}
	}

	// Capture UP path
	if (copy_playable_direction[i][j][1])
	{
		i_it = i - 1, j_it = j;
		while (copy_board[i_it][j_it] == opposing_player)
		{
			copy_board[i_it][j_it] = current_player;
			potential_score++;
			i_it -= 1;
		}
	}

	// Capture UR diagonal
	if (copy_playable_direction[i][j][2])
	{
		i_it = i - 1, j_it = j + 1;
		while (copy_board[i_it][j_it] == opposing_player)
		{
			copy_board[i_it][j_it] = curr_potential_player;
			potential_score++;
			i_it -= 1;
			j_it += 1;
		}
	}

	// Capture LEFT path
	if (copy_playable_direction[i][j][3])
	{
		i_it = i, j_it = j - 1;
		while (copy_board[i_it][j_it] == opposing_player)
		{
			copy_board[i_it][j_it] = curr_potential_player;
			potential_score++;
			j_it -= 1;
		}
	}

	// Capture RIGHT path
	if (copy_playable_direction[i][j][4])
	{
		i_it = i, j_it = j + 1;
		while (copy_board[i_it][j_it] == opposing_player)
		{
			copy_board[i_it][j_it] = curr_potential_player;
			potential_score++;
			j_it += 1;
		}
	}

	// Capture DL diagonal
	if (copy_playable_direction[i][j][5])
	{
		i_it = i + 1, j_it = j - 1;
		while (copy_board[i_it][j_it] == opposing_player)
		{
			copy_board[i_it][j_it] = curr_potential_player;
			potential_score++;
			i_it += 1;
			j_it -= 1;
		}
	}

	// Capture DOWN path
	if (copy_playable_direction[i][j][6])
	{
		i_it = i + 1, j_it = j;
		while (copy_board[i_it][j_it] == opposing_player)
		{
			copy_board[i_it][j_it] = curr_potential_player;
			potential_score++;
			i_it += 1;
		}
	}

	// Capture DR diagonal
	if (copy_playable_direction[i][j][7])
	{
		i_it = i + 1, j_it = j + 1;
		while (copy_board[i_it][j_it] == opposing_player)
		{
			copy_board[i_it][j_it] = curr_potential_player;
			potential_score++;
			i_it += 1;
			j_it += 1;
		}
	}
	return potential_score;
}

// Random player
void get_random_move_cpu(int* p_row, int* p_column)
{
	int count = 0;
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			if (board[i][j] == PLAYABLE) {
				count++;
				if (rand() % count == 0) {
					*p_row = i;
					*p_column = j;
				}
			}
		}
	}
}

void copy_board_mem_arr(char newBoard[8][8], int newPlayDir[8][8][8], char boardToCopy[8][8], int playableDirToCopy[8][8][8]){
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                newPlayDir[i][j][k] = playableDirToCopy[i][j][k];
            }
            newBoard[i][j] = boardToCopy[i][j];
        }
    }
}

// Think ahead 1 step
int get_easyAI_move_cpu(move* moves)
{
	int maxScore = 0;
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			if (board[i][j] == PLAYABLE) {
				char copy_board[8][8];
				int copy_playable_direction[8][8][8];
        copy_board_mem_arr(copy_board, copy_playable_direction, board, playable_direction);
				int s = capture_potential_pieces_cpu(i, j, copy_board, current_player, copy_playable_direction);
				moves[i*8 + j].max = s;
				moves[i*8 + j].i = i;
				moves[i*8 + j].j = j;
				if (s >= maxScore) {
					maxScore = s;
				}
			}
		}
	}
	return maxScore;
}

int predict_next_move_cpu(char cboard[8][8], int cplayable_direction[8][8][8], int tmp_current_player, int count) {
	if (count == 0) return 0;
	int maxScore = -100;
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			if (cboard[i][j] == PLAYABLE) {
				char copy_board[8][8];
				int copy_playable_direction[8][8][8];
        copy_board_mem_arr(copy_board, copy_playable_direction, cboard, cplayable_direction);
        copy_board[i][j] = tmp_current_player;
				int s = capture_potential_pieces_cpu(i, j, copy_board, tmp_current_player, copy_playable_direction);
				mark_playable_positions_simulate_cpu(copy_board, copy_playable_direction, (tmp_current_player + 1) % 2);
				int res = predict_next_move_cpu(copy_board, copy_playable_direction, (tmp_current_player + 1) % 2, count - 1);

				s = s - res;
				if (s >= maxScore) {
					maxScore = s;
				}
			}
		}
	}
	return maxScore;
}

// Think ahead 3 steps
int get_mediumAI_move_cpu(move* moves)
{
	int maxScore = -1000;
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			if (board[i][j] == PLAYABLE) {
				char copy_board[8][8];
				int copy_playable_direction[8][8][8];
        		copy_board_mem_arr(copy_board, copy_playable_direction, board, playable_direction);
        		copy_board[i][j] = current_player;
				int s = capture_potential_pieces_cpu(i, j, copy_board, current_player, copy_playable_direction);
				mark_playable_positions_simulate_cpu(copy_board, copy_playable_direction, (current_player + 1) % 2);
				int res = predict_next_move_cpu(copy_board, copy_playable_direction, (current_player + 1) % 2, STEPS_THINK_AHEAD-1);

				s = s - res;
				// printf("medium score: %d here, row: %d, column: %d\n", diff, i, j);
				moves[i * 8 + j].max = s;
				moves[i * 8 + j].i = i;
				moves[i * 8 + j].j = j;
				if (s >= maxScore) {
					maxScore = s;
				}
			}
		}
	}
	return maxScore;
}

void display_winner_cpu()
{
	printf("Final score:\n%s: %d %s: %d\n", WHITE_MARKER, scores[WHITE], BLACK_MARKER, scores[BLACK]);
	if (scores[WHITE] > scores[BLACK])
		printf("%s wins.\n", WHITE_MARKER);
	else if (scores[WHITE] < scores[BLACK])
		printf("%s wins.\n", BLACK_MARKER);
	else
		printf("Draw game.\n");
}

void display_score_cpu()
{
	printf("%s: %d %s: %d\n", WHITE_MARKER, scores[WHITE], BLACK_MARKER, scores[BLACK]);
}


////////////////////////////////////////////////////////////////////////////////// Call Kernel //////////////////////////////////////////////////////////////////////////////
void make_next_move()
{
	int row = 0;
	int column = 0;
  struct timeval t1, t2;
	if (AI_PLAYER == current_player) {
		if (ai == Easy_CUDA || ai == Medium_CUDA) {
			move* moves;
			char* devBoard;
			int* devplayDir;
			int m = 0;
			int n = 0;
			int max = -100;
			devBoard = (char*)malloc(64 * sizeof(char));
			devplayDir = (int*)malloc(512 * sizeof(int));
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					for (int k = 0; k < 8; k++)
					{
						devplayDir[n] = playable_direction[i][j][k];
						n++;
					}
					devBoard[m] = board[i][j];
					m++;
				}
			}
			char* devBoard2;
			int* devplayDir2;
			cudaMalloc(&devBoard2, 8 * 8 * sizeof(char));
			cudaMemcpy(devBoard2, devBoard, 8 * 8 * sizeof(char), cudaMemcpyHostToDevice);
			cudaMalloc(&devplayDir2, 8 * 8 * 8 * sizeof(int));
			cudaMemcpy(devplayDir2, devplayDir, 512 * sizeof(int), cudaMemcpyHostToDevice);
			cudaMallocManaged((void**)&moves, 64 * sizeof(move));
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					moves[i*8 + j].max = -200;
          moves[i*8 + j].i = i;
          moves[i*8 + j].j = j;
 				}
			}
      gettimeofday(&t1, NULL);
      if (ai == Easy_CUDA) {
        get_easyAI_move << < 1, 64 >> > (devBoard2, devplayDir2, current_player, moves);
      } else {
        get_mediumAI_move << < 1, 64 >> > (devBoard2, devplayDir2, current_player, moves);
      }
			cudaDeviceSynchronize();
      gettimeofday(&t2, NULL);
      total_move_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

			for (int i = 0; i < 64; i++)
			{
				if (moves[i].max > max)
				{
					max = moves[i].max;
				}
			}

			int sampling_count = 0;
			for (int i = 0; i < 64; i++)
			{
				if (moves[i].max == max)
				{
					sampling_count++;
					if (rand() % sampling_count == 0) {
						row = moves[i].i;
						column = moves[i].j;
					}
				}
			}
      //printf("max: %d\n", max);
			//printf("medium: row: %d, column: %d\n", row, column);
			cudaFree(devBoard2);
			cudaFree(devplayDir2);
			cudaFree(moves);
			free(devBoard);
			free(devplayDir);
		}
		else {
			move* moves;
			moves = (move*)malloc(64 * sizeof(move));
      for (int i = 0; i < 8; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					moves[i*8 + j].max = -200;
          moves[i*8 + j].i = i;
          moves[i*8 + j].j = j;
 				}
			}
      int max = -200;
      gettimeofday(&t1, NULL);
      if (ai == Easy) {
        max = get_easyAI_move_cpu(moves);
      } else {
        max = get_mediumAI_move_cpu(moves);
      }
      gettimeofday(&t2, NULL);
      total_move_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
			int sampling_count = 0;
			for (int i = 0; i < 64; i++)
			{
				if (moves[i].max == max)
				{
					sampling_count++;
					if (rand() % sampling_count == 0) {
						row = moves[i].i;
						column = moves[i].j;
					}
				}
			}
			free(moves);
		}
	}
	else {
    if (humanPlay == 0) {
			prompt_move_cpu( &row, &column );	
		} else {
			get_random_move_cpu(&row, &column);		
		}
		//printf("random: row: %d, column: %d\n", row, column);

		// Uncomment to have easy plays against the other ai
		/*
		move* moves;
		moves = (move*)malloc(64 * sizeof(move));
		int max = get_easyAI_move_cpu(moves);
		int sampling_count = 0;
		for (int i = 0; i < 64; i++)
		{
			if (moves[i].max == max)
			{
				sampling_count++;
				if (rand() % sampling_count == 0) {
					row = moves[i].i;
					column = moves[i].j;
				}
			}
		}
		free(moves);
		*/
	}
	if (is_valid_position_cpu(row, column) && board[row][column] == PLAYABLE)
	{
		// printf("row: %d, column: %d is valid\n", row, column);
		board[row][column] = current_player;
		scores[current_player]++;
		capture_pieces_cpu(row, column);
		change_current_player_cpu();
	}
	else wrong_move = TRUE;
}


int main(int argc, char *argv[])
{
  if (argc != 3)
  {
      fprintf(stderr, "%s <AI_Player Level> <AI play with human or not>\n", argv[0]);
      fprintf(stderr,"<AI_Player Level>: 0-Easy, 1-Easy_CUDA, 2-Medium, 3-Medium_CUDA\n");
      fprintf(stderr,"<AI play with human or not>: 0-Human plays with AI, 1-Random player plays with AI.\n");
      return 1;
  }

  ai = atoi(argv[1]);
  humanPlay = atoi(argv[2]);

	srand(time(NULL));
	int countXWin = 0;
	int draw = 0;
	int count0Win = 0;
	for (int i = 0; i < 100; i++) {
		init_game();
		while (!game_ended) {
			if (!wrong_move) mark_playable_positions_cpu();
			if (!has_valid_move)
			{
				if (skipped_turn)
				{
					game_ended = 1;
					//draw_board_cpu();
					continue;
				}
				skipped_turn = 1;
				change_current_player_cpu();
				continue;
			}
			skipped_turn = 0;
			if (humanPlay == 0) {
				draw_board_cpu( );
				display_score_cpu( );
				display_current_player_cpu( );		
			}
			
      if (wrong_move) {
          display_wrong_move_cpu( );
      }
			make_next_move();
		}
			if (humanPlay == 0) {
				draw_board_cpu( );		
			}
		display_winner_cpu();
		if (scores[WHITE] > scores[BLACK]) {
			countXWin++;
		}
		else if (scores[WHITE] < scores[BLACK]) {
			count0Win++;
		}
		else {
			draw++;
		}
	}
	printf("X wins : %d, 0 wins : %d, draw : %d\n", countXWin, count0Win, draw);
  printf("Total time that the AI spent to calculate moves: (%f ms)\n", total_move_time);
}