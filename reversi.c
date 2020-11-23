#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
#define STEPS_AHEAD 2
#define REPEAT_GAME 100

const char *row_names = "01234567";
const char *col_names = "01234567";

char board[8][8];
int playable_direction[8][8][8];
int current_player;
int game_ended = FALSE;
int skipped_turn = FALSE;
int wrong_move = FALSE;
int has_valid_move = FALSE;
int scores[2];
int black_score = 2;

void init_game( )
{
    // for(int i=0; i<8; i++){
    //     for(int j=0; j<8; j++){
    //         board[i][j] = EMPTY;
    //     }
    // }
    memset( board, EMPTY, sizeof( board ) );
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

int is_valid_position( int i, int j )
{
    if ( i < 0 || i >= 8 || j < 0 || j >= 8 ) return FALSE;
    return TRUE;
}

int distance( int i1, int j1, int i2, int j2 )
{
    int di = abs( i1 - i2 ), dj = abs( j1 - j2 );
    if ( di > 0 ) return di;
    return dj;
}

int is_playable( int i, int j )
{
   // memset( playable_direction[i][j], 0, 8 );
     for ( int m=0; m<8; ++m )
    {
      playable_direction[i][j][m] = 0;
    }
    if ( !is_valid_position( i, j ) ) return FALSE;
    if ( board[i][j] != EMPTY ) return FALSE;
    int playable = FALSE;

    int opposing_player = ( current_player + 1 ) % 2;
    
    // Test UL diagonal
    int i_it = i-1, j_it = j-1;
    while ( is_valid_position( i_it, j_it ) && board[i_it][j_it] == opposing_player )
    {
        i_it -= 1;
        j_it -= 1;
    }
    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && board[i_it][j_it] == current_player )
    {
        playable_direction[i][j][0] = 1;
        playable = TRUE;
    }

    // Test UP path
    i_it = i-1, j_it = j;
    while ( is_valid_position( i_it, j_it ) && board[i_it][j_it] == opposing_player )
        i_it -= 1;

    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && board[i_it][j_it] == current_player )
    {
        playable_direction[i][j][1] = 1;
        playable = TRUE;
    }
    
    // Test UR diagonal
    i_it = i-1, j_it = j+1;
    while ( is_valid_position( i_it, j_it ) && board[i_it][j_it] == opposing_player )
    {
        i_it -= 1;
        j_it += 1;
    }
    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && board[i_it][j_it] == current_player )
    {
        playable_direction[i][j][2] = 1;
        playable = TRUE;
    }

    // Test LEFT path
    i_it = i, j_it = j-1;
    while ( is_valid_position( i_it, j_it ) && board[i_it][j_it] == opposing_player )
        j_it -= 1;

    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && board[i_it][j_it] == current_player )
    {
        playable_direction[i][j][3] = 1;
        playable = TRUE;
    }

    // Test RIGHT path
    i_it = i, j_it = j+1;
    while ( is_valid_position( i_it, j_it ) && board[i_it][j_it] == opposing_player )
        j_it += 1;

    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && board[i_it][j_it] == current_player )
    {
        playable_direction[i][j][4] = 1;
        playable = TRUE;
    }

    // Test DL diagonal
    i_it = i+1, j_it = j-1;
    while ( is_valid_position( i_it, j_it ) && board[i_it][j_it] == opposing_player )
    {
        i_it += 1;
        j_it -= 1;
    }
    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && board[i_it][j_it] == current_player )
    {
       // printf("%d, %d, %d, %d\n", i, j, i_it, j_it);
        playable_direction[i][j][5] = 1;
        playable = TRUE;
    }

    // Test DOWN path
    i_it = i+1, j_it = j;
    while ( is_valid_position( i_it, j_it ) && board[i_it][j_it] == opposing_player )
        i_it += 1;

    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && board[i_it][j_it] == current_player )
    {
        playable_direction[i][j][6] = 1;
        playable = TRUE;
    }

    // Test DR diagonal
    i_it = i+1, j_it = j+1;
    while ( is_valid_position( i_it, j_it ) && board[i_it][j_it] == opposing_player )
    {
        i_it += 1;
        j_it += 1;
    }
    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && board[i_it][j_it] == current_player )
    {
        playable_direction[i][j][7] = 1;
        playable = TRUE;
    }
    return playable;
}

int is_playable_dummy( int i, int j, char copy_board[8][8], int copy_playable_direction[8][8][8], int tmp_current_player)
{
   // memset( copy_playable_direction[i][j], 0, 8 );
     for ( int m=0; m<8; ++m )
    {
      copy_playable_direction[i][j][m] = 0;
    }
    if ( !is_valid_position( i, j ) ) return FALSE;
    if ( copy_board[i][j] != EMPTY ) return FALSE;
    int playable = FALSE;

    int opposing_player = ( tmp_current_player + 1 ) % 2;
    
    // Test UL diagonal
    int i_it = i-1, j_it = j-1;
    while ( is_valid_position( i_it, j_it ) && copy_board[i_it][j_it] == opposing_player )
    {
        i_it -= 1;
        j_it -= 1;
    }
    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && copy_board[i_it][j_it] == tmp_current_player )
    {
        copy_playable_direction[i][j][0] = 1;
        playable = TRUE;
    }

    // Test UP path
    i_it = i-1, j_it = j;
    while ( is_valid_position( i_it, j_it ) && copy_board[i_it][j_it] == opposing_player )
        i_it -= 1;

    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && copy_board[i_it][j_it] == tmp_current_player )
    {
        copy_playable_direction[i][j][1] = 1;
        playable = TRUE;
    }
    
    // Test UR diagonal
    i_it = i-1, j_it = j+1;
    while ( is_valid_position( i_it, j_it ) && copy_board[i_it][j_it] == opposing_player )
    {
        i_it -= 1;
        j_it += 1;
    }
    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && copy_board[i_it][j_it] == tmp_current_player )
    {
        copy_playable_direction[i][j][2] = 1;
        playable = TRUE;
    }

    // Test LEFT path
    i_it = i, j_it = j-1;
    while ( is_valid_position( i_it, j_it ) && copy_board[i_it][j_it] == opposing_player )
        j_it -= 1;

    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && copy_board[i_it][j_it] == tmp_current_player )
    {
        copy_playable_direction[i][j][3] = 1;
        playable = TRUE;
    }

    // Test RIGHT path
    i_it = i, j_it = j+1;
    while ( is_valid_position( i_it, j_it ) && copy_board[i_it][j_it] == opposing_player )
        j_it += 1;

    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && copy_board[i_it][j_it] == tmp_current_player )
    {
        copy_playable_direction[i][j][4] = 1;
        playable = TRUE;
    }

    // Test DL diagonal
    i_it = i+1, j_it = j-1;
    while ( is_valid_position( i_it, j_it ) && copy_board[i_it][j_it] == opposing_player )
    {
        i_it += 1;
        j_it -= 1;
    }
    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && copy_board[i_it][j_it] == tmp_current_player )
    {
        copy_playable_direction[i][j][5] = 1;
        playable = TRUE;
    }

    // Test DOWN path
    i_it = i+1, j_it = j;
    while ( is_valid_position( i_it, j_it ) && copy_board[i_it][j_it] == opposing_player )
        i_it += 1;

    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && copy_board[i_it][j_it] == tmp_current_player )
    {
        copy_playable_direction[i][j][6] = 1;
        playable = TRUE;
    }

    // Test DR diagonal
    i_it = i+1, j_it = j+1;
    while ( is_valid_position( i_it, j_it ) && copy_board[i_it][j_it] == opposing_player )
    {
        i_it += 1;
        j_it += 1;
    }
    if ( is_valid_position( i_it, j_it ) && distance( i, j, i_it, j_it ) > 1 && copy_board[i_it][j_it] == tmp_current_player )
    {
        copy_playable_direction[i][j][7] = 1;
        playable = TRUE;
    }
    return playable;
}

void mark_playable_positions( )
{
    has_valid_move = FALSE;
    for ( int i=0; i<8; ++i )
    {
        for ( int j=0; j<8; ++j )
        {
            if ( board[i][j] == PLAYABLE )
                board[i][j] = EMPTY;
            if ( is_playable( i, j ) )
            {
                board[i][j] = PLAYABLE;
                has_valid_move = TRUE;
            }
        }
    }
}

void mark_playable_positions_dummy(char copy_board[8][8], int copy_playable_direction[8][8][8], int tmp_current_player)
{
    has_valid_move = FALSE;
    for ( int i=0; i<8; ++i )
    {
        for ( int j=0; j<8; ++j )
        {
            if ( copy_board[i][j] == PLAYABLE )
                copy_board[i][j] = EMPTY;
            if ( is_playable_dummy( i, j, copy_board, copy_playable_direction, tmp_current_player) )
            {
                copy_board[i][j] = PLAYABLE;
                has_valid_move = TRUE;
            }
        }
    }
}

void draw_board( )
{
    printf( "     %c     %c     %c     %c     %c     %c     %c     %c\n", col_names[0], col_names[1], col_names[2], col_names[3], col_names[4], col_names[5], col_names[6], col_names[7] );
    printf( "   _____ _____ _____ _____ _____ _____ _____ _____\n" );
    for ( int i=0; i<8; ++i )
    {
        printf( "  |     |     |     |     |     |     |     |     |\n" );
        printf( "%c |", row_names[i] );
        for ( int j=0; j<8; ++j )
        {
            if ( board[i][j] == WHITE )
            {
                printf( "%s", WHITE_MARKER );
            } else if ( board[i][j] == BLACK )
            {
                printf( "%s", BLACK_MARKER );
            } else if ( board[i][j] == PLAYABLE )
            {
                printf( "%s", PLAYABLE_MARKER );
            } else
            {
                printf( "%s", EMPTY_MARKER );
            }
            printf("|");
        }
        printf( "\n" );
        printf( "  |_____|_____|_____|_____|_____|_____|_____|_____|\n" );
    }
    printf( "\n" );
}

void display_wrong_move( )
{
    if ( wrong_move )
    {
        printf( "You entered an invalid move!\n" );
        wrong_move = FALSE;
    }
}

void display_current_player( )
{
    printf( "Current player:" );
    if ( current_player == WHITE )
        printf( "%s", WHITE_MARKER );
    else
        printf( "%s", BLACK_MARKER );
    printf( "\n" );
}

void change_current_player( )
{
    current_player = ( current_player + 1 ) % 2;
}

void prompt_move( int *p_row, int *p_column )
{
    printf( "Enter row [0-7] and column [0-7] separated by a single space (eg.: 2 4).\n" );
    scanf( "%d %d", p_row, p_column );
}

void capture_pieces( int i, int j )
{
    int opposing_player = ( current_player + 1 ) % 2;
    int i_it, j_it;
    //printf("capture_pieces: row: %d, column: %d\n", i, j);
    // Capture UL diagonal
    if ( playable_direction[i][j][0] )
    {
        i_it = i-1, j_it = j-1;
        while ( board[i_it][j_it] == opposing_player )
        {
            board[i_it][j_it] = current_player;
            scores[current_player]++;
            scores[opposing_player]--;
            i_it -= 1;
            j_it -= 1;
            if(scores[opposing_player] < 0) printf("problem 1");
        }
    }

    // Capture UP path
    if ( playable_direction[i][j][1] )
    {
        i_it = i-1, j_it = j;
        while ( board[i_it][j_it] == opposing_player )
        {
            board[i_it][j_it] = current_player;
            scores[current_player]++;
            scores[opposing_player]--;
            i_it -= 1;
            if(scores[opposing_player] < 0) printf("problem 2");
        }
    }
    
    // Capture UR diagonal
    if ( playable_direction[i][j][2] )
    {
        i_it = i-1, j_it = j+1;
        while ( board[i_it][j_it] == opposing_player )
        {
            board[i_it][j_it] = current_player;
            scores[current_player]++;
            scores[opposing_player]--;
            i_it -= 1;
            j_it += 1;
            if(scores[opposing_player] < 0) printf("problem 3");
        }
    }

    // Capture LEFT path
    if ( playable_direction[i][j][3] )
    {
        i_it = i, j_it = j-1;
        while ( board[i_it][j_it] == opposing_player )
        {
            board[i_it][j_it] = current_player;
            scores[current_player]++;
            scores[opposing_player]--;
            j_it -= 1;
            if(scores[opposing_player] < 0) printf("problem 4");
        }
    }

    // Capture RIGHT path
    if ( playable_direction[i][j][4] )
    {
        i_it = i, j_it = j+1;
        while ( board[i_it][j_it] == opposing_player )
        {
            board[i_it][j_it] = current_player;
            scores[current_player]++;
            scores[opposing_player]--;
            j_it += 1;
            if(scores[opposing_player] < 0) printf("problem 5");
        }
    }

    // Capture DL diagonal
    if ( playable_direction[i][j][5] )
    {
        i_it = i+1, j_it = j-1;
        while ( board[i_it][j_it] == opposing_player )
        {
            board[i_it][j_it] = current_player;
            scores[current_player]++;
            scores[opposing_player]--;
            i_it += 1;
            j_it -= 1;
            if(scores[opposing_player] < 0) printf("problem 6\n");
        }
    }

    // Capture DOWN path
    if ( playable_direction[i][j][6] )
    {
        i_it = i+1, j_it = j;
        while ( board[i_it][j_it] == opposing_player )
        {
            board[i_it][j_it] = current_player;
            scores[current_player]++;
            scores[opposing_player]--;
            i_it += 1;
            if(scores[opposing_player] < 0) printf("problem 7");
        }
    }

    // Capture DR diagonal
    if ( playable_direction[i][j][7] )
    {
        i_it = i+1, j_it = j+1;
        while ( board[i_it][j_it] == opposing_player )
        {
            board[i_it][j_it] = current_player;
            scores[current_player]++;
            scores[opposing_player]--;
            i_it += 1;
            j_it += 1;
            if(scores[opposing_player] < 0) printf("problem 8");
        }
    }
}

int capture_potential_pieces( int i, int j, char copy_board[8][8], int curr_potential_player, int copy_playable_direction[8][8][8])
{
    int opposing_player = ( curr_potential_player + 1 ) % 2;
    int potential_score = 0;
    int i_it, j_it;
    
    // Capture UL diagonal
    if ( copy_playable_direction[i][j][0] )
    {
        i_it = i-1, j_it = j-1;
        while ( copy_board[i_it][j_it] == opposing_player )
        {
            copy_board[i_it][j_it] = curr_potential_player;
            potential_score++;
            i_it -= 1;
            j_it -= 1;
        }
    }

    // Capture UP path
    if ( copy_playable_direction[i][j][1] )
    {
        i_it = i-1, j_it = j;
        while ( copy_board[i_it][j_it] == opposing_player )
        {
            copy_board[i_it][j_it] = current_player;
            potential_score++;
            i_it -= 1;
        }
    }
    
    // Capture UR diagonal
    if ( copy_playable_direction[i][j][2] )
    {
        i_it = i-1, j_it = j+1;
        while ( copy_board[i_it][j_it] == opposing_player )
        {
            copy_board[i_it][j_it] = curr_potential_player;
            potential_score++;
            i_it -= 1;
            j_it += 1;
        }
    }

    // Capture LEFT path
    if ( copy_playable_direction[i][j][3] )
    {
        i_it = i, j_it = j-1;
        while ( copy_board[i_it][j_it] == opposing_player )
        {
            copy_board[i_it][j_it] = curr_potential_player;
            potential_score++;
            j_it -= 1;
        }
    }

    // Capture RIGHT path
    if ( copy_playable_direction[i][j][4] )
    {
        i_it = i, j_it = j+1;
        while ( copy_board[i_it][j_it] == opposing_player )
        {
            copy_board[i_it][j_it] = curr_potential_player;
            potential_score++;
            j_it += 1;
        }
    }

    // Capture DL diagonal
    if ( copy_playable_direction[i][j][5] )
    {
        i_it = i+1, j_it = j-1;
        while ( copy_board[i_it][j_it] == opposing_player )
        {
            copy_board[i_it][j_it] = curr_potential_player;
            potential_score++;
            i_it += 1;
            j_it -= 1;
        }
    }

    // Capture DOWN path
    if ( copy_playable_direction[i][j][6] )
    {
        i_it = i+1, j_it = j;
        while ( copy_board[i_it][j_it] == opposing_player )
        {
            copy_board[i_it][j_it] = curr_potential_player;
            potential_score++;
            i_it += 1;
        }
    }

    // Capture DR diagonal
    if ( copy_playable_direction[i][j][7] )
    {
        i_it = i+1, j_it = j+1;
        while ( copy_board[i_it][j_it] == opposing_player )
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
void get_random_move( int *p_row, int *p_column )
{

  int count = 0;
  for ( int i=0; i<8; ++i )
  {
    for ( int j=0; j<8; ++j )
    {
      if (board[i][j] == PLAYABLE) {
        count++;
        if (rand() % count == 0) {
            printf("random: %d", rand());
          *p_row = i;
          *p_column = j;
        }        
      }
    }
  }
}

// Think ahead 1 step
void get_easyAI_move( int *p_row, int *p_column )
{
  int maxScore = 0;
  for ( int i=0; i<8; ++i )
    {
        for ( int j=0; j<8; ++j )
        {
          if (board[i][j] == PLAYABLE) {
              char copy_board[8][8];
              int copy_playable_direction[8][8][8];
              memcpy(copy_board, board, sizeof (char) * 8 * 8);
              memcpy(copy_playable_direction, playable_direction, sizeof (int) * 8 * 8 * 8);
              int s = capture_potential_pieces(i, j, copy_board, current_player, copy_playable_direction);
              if (s >= maxScore) {
                maxScore = s;
                *p_row = i;
                *p_column = j;
                //printf("current player: %d here\n", current_player);
              }
          }
        }
    }
}

int predict_next_move(char cboard[8][8], int cplayable_direction[8][8][8], int tmp_current_player, int count){
    if(count == 0) return 0;
    int maxScore = -100;
    for ( int i=0; i<8; ++i ) {
        for ( int j=0; j<8; ++j ) {
          if (cboard[i][j] == PLAYABLE) {
              char copy_board[8][8];
              int copy_playable_direction[8][8][8];
              memcpy(copy_board, cboard, sizeof (char) * 8 * 8);
              memcpy(copy_playable_direction, cplayable_direction, sizeof (int) * 8 * 8 * 8);

              int s = capture_potential_pieces(i, j, copy_board, tmp_current_player, copy_playable_direction);

              mark_playable_positions_dummy(copy_board, copy_playable_direction, (tmp_current_player+1) % 2);
              int res = predict_next_move(copy_board, copy_playable_direction, (tmp_current_player+1) % 2, count-1);

              s = s - res;
              if (s >= maxScore) {
                maxScore = s;
              }
          }
        }
    }
    if(maxScore == -100) maxScore = 0;
    return maxScore;
}

// Think ahead 3 steps
void get_mediumAI_move( int *p_row, int *p_column )
{
  int maxScore = -1000;
  for ( int i=0; i<8; ++i )
    {
        for ( int j=0; j<8; ++j )
        {
          if (board[i][j] == PLAYABLE) {
              char copy_board[8][8];
              int copy_playable_direction[8][8][8];
              memcpy(copy_board, board, sizeof (char) * 8 * 8);
              memcpy(copy_playable_direction, playable_direction, sizeof (int) * 8 * 8 * 8);
              int s = capture_potential_pieces(i, j, copy_board, current_player, copy_playable_direction);
              mark_playable_positions_dummy(copy_board, copy_playable_direction, (current_player+1) % 2);
              int res = predict_next_move(copy_board, copy_playable_direction, (current_player+1) % 2, STEPS_AHEAD);

              int diff = s - res;
             // printf("medium score: %d here, row: %d, column: %d\n", diff, i, j);
              if (diff >= maxScore) {
                maxScore = s;
                *p_row = i;
                *p_column = j;
               // printf("maxScore: %d here\n", maxScore);
              }
          }
        }
    }
}

void make_next_move( )
{
    int row, column;
    if (AI_PLAYER == current_player) {
      get_random_move( &row, &column );
      //printf("random: row: %d, column: %d\n", row, column);
    } else {
       //prompt_move( &row, &column );
       get_mediumAI_move( &row, &column );
     //  printf("easy: row: %d, column: %d\n", row, column);
    }
    if ( is_valid_position( row, column ) && board[row][column] == PLAYABLE )
    {
       // printf("row: %d, column: %d is valid\n", row, column);
        board[row][column] = current_player;
        scores[current_player]++;
        capture_pieces( row, column );
        change_current_player(  );
    }
    else wrong_move = TRUE;
}

void display_winner( )
{
    printf( "Final score:\n%s: %d %s: %d\n", WHITE_MARKER, scores[WHITE], BLACK_MARKER, scores[BLACK] );
    if ( scores[WHITE] > scores[BLACK] )
        printf( "%s wins.\n", WHITE_MARKER );
    else if ( scores[WHITE] < scores[BLACK] )
        printf( "%s wins.\n", BLACK_MARKER );
    else
        printf( "Draw game.\n" );
}

void display_score( )
{
    printf( "%s: %d %s: %d\n", WHITE_MARKER, scores[WHITE], BLACK_MARKER, scores[BLACK] );
}

int main( )
{
    srand(time(NULL));
    int countXWin = 0;
    int draw = 0;
    int count0Win = 0;
    for(int i=0; i<REPEAT_GAME; i++){
        init_game();
        while ( !game_ended ){
            if ( !wrong_move ) mark_playable_positions( );
            if ( !has_valid_move )
            {
                if ( skipped_turn )
                {

                    game_ended = 1;
                    draw_board( );
                    continue;
                }
                skipped_turn = 1;
                change_current_player( );
                continue;
            }
            skipped_turn = 0;
            //draw_board( );
            //display_score( );
            //display_current_player( );
            //display_wrong_move( );
            if(wrong_move) exit(0);
            make_next_move( );
        }
        //mark_playable_positions();
       // draw_board( );
        display_winner( );
        if(scores[WHITE] > scores[BLACK]){
            countXWin++;
        } else if(scores[WHITE] < scores[BLACK]){
            count0Win++;
        } else {
            draw++;
        }
    }

    printf("x wins : %d, 0 wins : %d, draw : %d\n", countXWin, count0Win, draw);
}