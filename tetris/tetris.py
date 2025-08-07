import pygame
import random
import sys

pygame.init()
GRID_WIDTH = 10
GRID_HEIGHT = 20
CELL_SIZE = 3
SCREEN_WIDTH = 64
SCREEN_HEIGHT = 64
GAME_OFFSET_X = 9
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)

TETRIS_SHAPES = [
    [['.....',
      '..#..',
      '..#..',
      '..#..',
      '..#..'],
     ['.....',
      '.....',
      '####.',
      '.....',
      '.....']],
    [['.....',
      '.....',
      '.##..',
      '.##..',
      '.....']],
    [['.....',
      '.....',
      '.#...',
      '###..',
      '.....'],
     ['.....',
      '.....',
      '.#...',
      '.##..',
      '.#...'],
     ['.....',
      '.....',
      '.....',
      '###..',
      '.#...'],
     ['.....',
      '.....',
      '.#...',
      '##...',
      '.#...']],
    [['.....',
      '.....',
      '.##..',
      '##...',
      '.....'],
     ['.....',
      '.#...',
      '.##..',
      '..#..',
      '.....']],
    [['.....',
      '.....',
      '##...',
      '.##..',
      '.....'],
     ['.....',
      '..#..',
      '.##..',
      '.#...',
      '.....']],
    [['.....',
      '.#...',
      '.#...',
      '##...',
      '.....'],
     ['.....',
      '.....',
      '#....',
      '###..',
      '.....'],
     ['.....',
      '.##..',
      '.#...',
      '.#...',
      '.....'],
     ['.....',
      '.....',
      '###..',
      '..#..',
      '.....']],
    [['.....',
      '..#..',
      '..#..',
      '.##..',
      '.....'],
     ['.....',
      '.....',
      '###..',
      '#....',
      '.....'],
     ['.....',
      '##...',
      '.#...',
      '.#...',
      '.....'],
     ['.....',
      '.....',
      '..#..',
      '###..',
      '.....']]
]

SHAPE_COLORS = [CYAN, YELLOW, PURPLE, GREEN, RED, BLUE, ORANGE]

class Piece:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.shape = random.randint(0, len(TETRIS_SHAPES) - 1)
        self.color = SHAPE_COLORS[self.shape]
        self.rotation = 0
    
    def image(self):
        return TETRIS_SHAPES[self.shape][self.rotation]
    
    def rotate(self):
        self.rotation = (self.rotation + 1) % len(TETRIS_SHAPES[self.shape])

class Tetris:
    def __init__(self):
        self.reset_game()
        
    def reset_game(self):
        self.grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.score = 0
        self.lines_cleared = 0
        self.fall_time = 0
        self.fall_speed = 200
        
    def new_piece(self):
        return Piece(GRID_WIDTH // 2 - 2, 0)
    
    def valid_move(self, piece, dx, dy, rotation=None):
        if rotation is None:
            rotation = piece.rotation
        
        shape = TETRIS_SHAPES[piece.shape][rotation]
        
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell == '#':
                    nx, ny = piece.x + x + dx, piece.y + y + dy
                    if (nx < 0 or nx >= GRID_WIDTH or 
                        ny >= GRID_HEIGHT or 
                        (ny >= 0 and self.grid[ny][nx] != BLACK)):
                        return False
        return True
    
    def place_piece(self, piece):
        shape = piece.image()
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell == '#':
                    nx, ny = piece.x + x, piece.y + y
                    if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH:
                        self.grid[ny][nx] = piece.color
    
    def clear_lines(self):
        lines_to_clear = []
        for y in range(GRID_HEIGHT):
            if all(cell != BLACK for cell in self.grid[y]):
                lines_to_clear.append(y)
        
        for y in lines_to_clear:
            del self.grid[y]
            self.grid.insert(0, [BLACK for _ in range(GRID_WIDTH)])
        
        lines_cleared = len(lines_to_clear)
        self.lines_cleared += lines_cleared
        self.score += lines_cleared * 100
        self.fall_speed = max(50, 500)
    
    def game_over(self):
        return not self.valid_move(self.current_piece, 0, 0)
    
    def update(self, dt):
        if self.valid_move(self.current_piece, 0, 1):
            self.current_piece.y += 1
        else:
            self.place_piece(self.current_piece)
            self.clear_lines()
            self.current_piece = self.next_piece
            self.next_piece = self.new_piece()
    
    def move_piece(self, dx):
        if self.valid_move(self.current_piece, dx, 0):
            self.current_piece.x += dx
    
    def rotate_piece(self):
        new_rotation = (self.current_piece.rotation + 1) % len(TETRIS_SHAPES[self.current_piece.shape])
        if self.valid_move(self.current_piece, 0, 0, new_rotation):
            self.current_piece.rotation = new_rotation
    
    def drop_piece(self):
        while self.valid_move(self.current_piece, 0, 1):
            self.current_piece.y += 1
        self.score += 2
    
    def draw(self, screen):
        screen.fill(BLACK)
        
        game_area_width = GRID_WIDTH * CELL_SIZE
        game_area_height = GRID_HEIGHT * CELL_SIZE
        pygame.draw.rect(screen, WHITE, 
                        (GAME_OFFSET_X - 1, 0 - 1, game_area_width + 2, game_area_height + 2), 1)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                color = self.grid[y][x]
                if color != BLACK:
                    pygame.draw.rect(screen, color, 
                                   (GAME_OFFSET_X + x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        if self.current_piece:
            shape = self.current_piece.image()
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell == '#':
                        nx, ny = self.current_piece.x + x, self.current_piece.y + y
                        if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH:
                            pygame.draw.rect(screen, self.current_piece.color,
                                           (GAME_OFFSET_X + nx * CELL_SIZE, ny * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        if self.next_piece:
            right_area_start = GAME_OFFSET_X + game_area_width + 2
            right_area_width = SCREEN_WIDTH - right_area_start
            
            next_shape = self.next_piece.image()
            shape_width = 0
            shape_height = 0
            for row in next_shape:
                if '#' in row:
                    shape_height += 1
                    shape_width = max(shape_width, len(row.rstrip('.')))
            shape_pixel_width = shape_width * 3
            shape_pixel_height = shape_height * 3
            
            preview_x = right_area_start + (right_area_width - shape_pixel_width) // 2
            preview_y = (SCREEN_HEIGHT - shape_pixel_height) // 2 - 12
            current_y = 0
            for y, row in enumerate(next_shape):
                if '#' in row:
                    current_x = 0
                    for x, cell in enumerate(row):
                        if cell == '#':
                            px = preview_x + current_x * 3
                            py = preview_y + current_y * 3
                            if px >= 0 and py >= 0 and px < SCREEN_WIDTH - 3 and py < SCREEN_HEIGHT - 3:
                                pygame.draw.rect(screen, self.next_piece.color,
                                               (px, py, 3, 3))
                        if x < shape_width:
                            current_x += 1
                    current_y += 1
        

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("俄罗斯方块 64x64")
    clock = pygame.time.Clock()
    
    game = Tetris()
    
    running = True
    while running:
        dt = clock.tick(10)  # 设置为10FPS，每帧下落一次
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.move_piece(-1)
                elif event.key == pygame.K_RIGHT:
                    game.move_piece(1)
                elif event.key == pygame.K_DOWN:
                    if game.valid_move(game.current_piece,0 ,1):
                        game.current_piece.y += 1;
                elif event.key == pygame.K_UP:
                    game.rotate_piece()
                elif event.key == pygame.K_SPACE:
                    game.drop_piece()
                elif event.key == pygame.K_q:
                    running = False
        
        if not game.game_over():
            game.update(dt)
        else:
            game.reset_game()
        
        game.draw(screen)
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
