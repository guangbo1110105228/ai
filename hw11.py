# # 參考https://www.jb51.net/article/206750.htm與GPT
import numpy as np

# 棋盤大小
ROW_COUNT = 6
COLUMN_COUNT = 7

# 玩家和電腦的棋子標記
PLAYER_PIECE = 1
AI_PIECE = 2

# 評估函數中用到的連線成績
SCORES = {0: 0,
          1: 1,
          2: 10,
          3: 100,
          4: 1000}


def create_board():
    """ 建立空的棋盤 """
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)


def drop_piece(board, row, col, piece):
    """ 在指定位置落子 """
    board[row][col] = piece


def is_valid_location(board, col):
    """ 檢查指定的列是否可以落子 """
    return board[ROW_COUNT-1][col] == 0


def get_next_open_row(board, col):
    """ 取得指定列中下一個可落子的位置 """
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def winning_move(board, piece):
    """ 檢查是否有連成四子的局面 """
    # 檢查水平方向
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # 檢查垂直方向
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # 檢查右上到左下的斜線方向
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # 檢查左上到右下的斜線方向
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

    return False


def evaluate_window(window, piece):
    """ 評估每個窗口的成績 """
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    score = 0
    if window.count(piece) == 4:
        score += SCORES[4]
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += SCORES[3]
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += SCORES[2]
    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= SCORES[3]
    return score


def score_position(board, piece):
    """ 評估當前局面的成績 """
    score = 0

    # 中心列優先考慮
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 6

    # 水平方向評估
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    # 垂直方向評估
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    # 右上到左下斜線方向評估
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # 左上到右下斜線方向評估
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    """ 判斷當前局面是否為終結局面 """
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizing_player):
    """ Minimax算法與alpha-beta剪枝 """
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:  # 沒有位置可以下了
                return (None, 0)
        else:  # Depth == 0
            return (None, score_position(board, AI_PIECE))

    if maximizing_player:
        value = -np.Inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value = np.Inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def get_valid_locations(board):
    """ 取得當前可以落子的位置 """
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def print_board(board):
    """ 將棋盤列印出來 """
    print(np.flip(board, 0))


def main():
    board = create_board()
    game_over = False
    turn = 0

    while not game_over:
        # 玩家回合
        if turn == 0:
            col = int(input("請輸入要落子的列 (0-6):"))
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                if winning_move(board, PLAYER_PIECE):
                    print("恭喜！你贏了！")
                    game_over = True

                turn += 1
                print_board(board)

        # AI回合
        if turn == 1 and not game_over:
            col, minimax_score = minimax(board, 5, -np.Inf, np.Inf, True)

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    print("電腦贏了！")
                    game_over = True

                print_board(board)
                turn -= 1

        if len(get_valid_locations(board)) == 0:
            print("平局！")
            game_over = True


if __name__ == "__main__":
    main()