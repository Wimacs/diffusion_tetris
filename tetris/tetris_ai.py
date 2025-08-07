import pygame
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tetris import Tetris, Piece, GRID_WIDTH, GRID_HEIGHT, TETRIS_SHAPES, BLACK, SCREEN_WIDTH, SCREEN_HEIGHT

class TetrisAI:
    def __init__(self, game):
        self.game = game
        self.move_sequence = []
        self.is_thinking = False
        self.thinking_time = 0
        

        self.mode = random.choice([1, 1,1, 2, 3])
        if self.mode == 1:
            self.mode_duration = 300
            print("AI Mode switched to: SMART")
        elif self.mode == 2:
            self.mode_duration = 150
            print("AI Mode switched to: PASSIVE")
        elif self.mode == 3:
            self.mode_duration = 150
            print("AI Mode switched to: FOOL")
        self.mode_frame_counter = 0

    def switch_mode(self):
        self.mode = random.choice([1, 1,1, 2, 3])
        if self.mode == 1:
            self.mode_duration = 300
            print("AI Mode switched to: SMART")
        elif self.mode == 2:
            self.mode_duration = 150
            print("AI Mode switched to: PASSIVE")
        elif self.mode == 3:
            self.mode_duration = 150
            print("AI Mode switched to: FOOL")
        
        # 重置状态
        self.mode_frame_counter = 0
        self.move_sequence = []
        self.is_thinking = False

    def update(self, dt):
        self.mode_frame_counter += 1
        if self.mode_frame_counter >= self.mode_duration:
            self.switch_mode()

        if self.mode == 1:
            return self.update_smart(dt)
        elif self.mode == 2:
            return self.update_passive()
        elif self.mode == 3:
            return self.update_fool()
        return 0

    def update_smart(self, dt):
        if self.is_thinking:
            self.thinking_time -= dt
            if self.thinking_time <= 0:
                self.is_thinking = False
                best_move = self.find_best_move()
                self.get_move_sequence(best_move)
            return 0  # 思考中，无操作

        action_code = 0
        if self.move_sequence:
            move = self.move_sequence.pop(0)
            if move == 'pause': action_code = 0
            elif move == 'left':
                self.game.move_piece(-1); action_code = 3
            elif move == 'right':
                self.game.move_piece(1); action_code = 4
            elif move == 'rotate':
                self.game.rotate_piece(); action_code = 1
            elif move == 'drop':
                self.game.drop_piece(); action_code = 2
        return action_code

    def update_passive(self):
        return 0
    
    def update_fool(self):
        action = random.choice([0, 1, 2, 3, 4]) # 0-无, 1-上, 2-下, 3-左, 4-右
        if action == 1: self.game.rotate_piece()
        elif action == 2:
            if self.game.valid_move(self.game.current_piece, 0, 1):
                self.game.current_piece.y += 1
        elif action == 3: self.game.move_piece(-1)
        elif action == 4: self.game.move_piece(1)
        return action
    
    def evaluate_grid(self, grid):
        heights = [0] * GRID_WIDTH
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if grid[y][x] != BLACK:
                    heights[x] = GRID_HEIGHT - y
                    break
        
        aggregate_height = sum(heights)
        max_height = max(heights)
        
        holes = 0
        for x in range(GRID_WIDTH):
            block_found = False
            for y in range(GRID_HEIGHT):
                if grid[y][x] != BLACK:
                    block_found = True
                elif block_found and grid[y][x] == BLACK:
                    holes += 1
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i+1])
            
        lines_cleared = 0
        for y in range(GRID_HEIGHT):
            if all(cell != BLACK for cell in grid[y]):
                lines_cleared += 1

        score = 0
        score -= aggregate_height * 0.5
        score -= holes * 1.0
        score -= bumpiness * 0.2
        score += lines_cleared * 10.0
        score -= max_height * 0.3
        
        return score

    def find_best_move(self):
        best_score = -float('inf')
        best_move = None
        
        current_piece = self.game.current_piece
        
        for rotation in range(len(TETRIS_SHAPES[current_piece.shape])):
            for x in range(-2, GRID_WIDTH + 2):
                
                test_piece = Piece(x, current_piece.y)
                test_piece.shape = current_piece.shape
                test_piece.rotation = rotation

                if not self.game.valid_move(test_piece, 0, 0):
                    continue

                y = test_piece.y
                while self.game.valid_move(test_piece, 0, 1):
                    test_piece.y += 1
                
                temp_grid = [row[:] for row in self.game.grid]
                
                shape = test_piece.image()
                for py, row in enumerate(shape):
                    for px, cell in enumerate(row):
                        if cell == '#':
                            ny, nx = test_piece.y + py, test_piece.x + px
                            if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH:
                                temp_grid[ny][nx] = test_piece.color
                
                score = self.evaluate_grid(temp_grid)
                
                if score > best_score:
                    best_score = score
                    best_move = (rotation, test_piece.x)

        return best_move

    def get_move_sequence(self, best_move):
        if best_move is None:
            self.move_sequence = ['drop']
            return

        target_rotation, target_x = best_move
        
        sequence = []

        if random.random() < 0.15:
            sequence.extend(['rotate', 'pause', 'rotate', 'pause', 'rotate', 'pause', 'rotate', 'pause'])

        current_rotation = self.game.current_piece.rotation
        rotations_needed = (target_rotation - current_rotation + 4) % 4
        for _ in range(rotations_needed):
            sequence.append('rotate')
            if random.random() < 0.5:
                sequence.append('pause')

        dx = target_x - self.game.current_piece.x
        move_cmd = 'right' if dx > 0 else 'left'
        for _ in range(abs(dx)):
            sequence.append(move_cmd)
            if random.random() < 0.5:
                sequence.append('pause')
            
        sequence.append('drop')
        
        self.move_sequence = sequence

    def start_thinking(self):
        if self.mode == 1:
            self.move_sequence = []
            self.is_thinking = True
            self.thinking_time = random.randint(100, 400)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris AI")
    clock = pygame.time.Clock()
    
    DATA_DIR = "data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    action_file_path = os.path.join(DATA_DIR, "action.txt")
    with open(action_file_path, "w") as f:
        pass
    
    frame_number = 0
    font = pygame.font.Font(None, 18) 

    game = Tetris()
    ai = TetrisAI(game)
    
    running = True
    auto_mode = True 
    
    last_piece = None

    while running:
        dt = clock.tick(15) 
        current_action = 0
        
        if game.current_piece != last_piece:
            last_piece = game.current_piece
            if auto_mode:
                ai.start_thinking()
            else:
                ai.move_sequence = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_mode = not auto_mode
                elif event.key == pygame.K_q:
                    running = False
                
                if not auto_mode:
                    if event.key == pygame.K_LEFT:
                        game.move_piece(-1)
                        current_action = 3
                    elif event.key == pygame.K_RIGHT:
                        game.move_piece(1)
                        current_action = 4
                    elif event.key == pygame.K_DOWN:
                        game.score += 1
                        if game.valid_move(game.current_piece,0 ,1):
                            game.current_piece.y += 1
                        current_action = 2
                    elif event.key == pygame.K_UP:
                        game.rotate_piece()
                        current_action = 1
        
        if not game.game_over():
            if auto_mode:
                current_action = ai.update(dt)
            
            game.update(dt)
            
        else:
            game.reset_game()
            ai = TetrisAI(game)
            last_piece = None
        
        game.draw(screen)

        #mode_str = {1: "SMART", 2: "PASSIVE", 3: "FOOL"}.get(ai.mode, "UNKNOWN")
        #if not auto_mode: mode_str = "MANUAL"
        #mode_text = font.render(mode_str, True, (255, 255, 255))
        #screen.blit(mode_text, (SCREEN_WIDTH - 55, 5))
        

        image_path = os.path.join(DATA_DIR, f"{frame_number}.png")
        pygame.image.save(screen, image_path)

        with open(action_file_path, "a") as f:
            f.write(f"{current_action}\n")
        
        frame_number += 1
                
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 