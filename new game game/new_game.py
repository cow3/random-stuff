#initialize game
import pygame
pygame.init()
#pygame.mixer.music.load('element.mp3')
#pygame.mixer.music.play(1000)
playing = True
pygame.display.set_caption('new game')
width = 1200
height = 720
black = (0,0,0)
white = (255,255,255)
backcolour = (0, 200 ,255)
background = pygame.transform.scale(pygame.image.load('backimg.png'), (width, height))
window = pygame.display.set_mode((width,height))
window.fill(backcolour)
pygame.display.update()
clock = pygame.time.Clock()
keys = ((pygame.K_a, pygame.K_d, pygame.K_s, pygame.K_w), (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_UP))

#functions and classes
def text(string, location = [800,400], size = 115):
    str1 = pygame.font.Font('freesansbold.ttf',size).render(string, True, black)
    strec = str1.get_rect()
    strec.center = location
    window.blit(str1,strec)

bound_int = lambda x,y:max(min(y), min(max(y), x))

def move_characters(players):
    #find current position
    players[0].relative_pos[0] = players[0].location[0] - players[1].location[0]
    players[1].relative_pos[0] = -players[0].relative_pos[0]
    players[0].relative_pos[1] = players[0].location[1] - players[1].location[1]
    players[1].relative_pos[1] = -players[0].relative_pos[1]
    assert not (-80 < players[0].relative_pos[0] < 80 and -120 < players[0].relative_pos[1] < 120)
    
    #vertical
    for i in 0,1:
        players[i].speed[1] += players[i].acceleration[1]
        players[i].location[1] = bound_int(players[i].location[1] + players[i].speed[1],[200, 600])
        if players[i].location[1] == 200:
            players[i].speed[1] = 0
               
    if -80 < players[0].relative_pos[0] < 80:
        t = (0 < players[1].relative_pos[1]);b = 1 - t
        if players[t].location[1] < 120 + players[b].location[1]:
            players[t].location[1] = players[b].location[1] + 125
            players[t].speed[1] = max(players[t].speed[1], 18)
            players[b].speed[1] = min(players[b].speed[1], 0)
            
    players[0].relative_pos[1] = players[0].location[1] - players[1].location[1]
    players[1].relative_pos[1] = -players[0].relative_pos[1]
    assert not (-80 < players[0].relative_pos[0] < 80 and -120 < players[0].relative_pos[1] < 120)
    #horizontal
    for i in 0,1:
        players[i].speed[0] = bound_int(players[i].speed[0] - players[i].acceleration[0], [-players[i].max_speed, players[i].max_speed])
        players[i].location[0] -= players[i].speed[0]
        if players[i].location[0] < 0 or players[i].location[0] > width - 120:
            players[i].location[0] += players[i].speed[0]
            players[i].speed[0] = -players[i].speed[0]
               
    if -120 < players[0].relative_pos[1] < 120:
        r = (0 < players[1].relative_pos[0]);l = 1 - r
        if  players[r].location[0] < players[l].location[0] + 80:
            if players[r].location[0] < width/2: 
                players[r].location[0] = players[l].location[0] + 80
            else:
                players[l].location[0] = players[r].location[0] - 80
            avspeed = (players[r].speed[0] + players[l].speed[0])/2
            players[r].speed[0] = avspeed
            players[l].speed[0] = avspeed
    
    for i in 0,1:
        if players[i].speed[0] != 0:
            players[i].facing_right = players[i].speed[0] < 0
        players[i].draw()

class character:
    def __init__(self, sprites, gravity = 1,max_speed = 12, can_fly = False, jump_speed = 23, location = [width * 0.45, height * 0.5], dmg = 5, current_sprite = 0):
        self.left_img = pygame.transform.scale(pygame.image.load(sprites[current_sprite]), (120, 160))
        self.right_img = pygame.transform.flip(self.left_img, 1, 0)
        self.health = 1000
        self.location = [width * 0.45, height * 0.5]
        self.facing_right = 0
        self.max_speed = max_speed
        self.name = sprites[0][:sprites[0].index(".")]
        self.acceleration = [0, -gravity]
        self.speed = [0, 0]
        self.relative_pos = [0, 0]
        self.sprites = sprites
        self.can_fly = can_fly
        self.jump_speed = jump_speed
        self.dmg = dmg
        self.current_sprite = current_sprite
        
    def sprite_change(self, amount = 1):
        self.current_sprite =  (self.current_sprite + amount) % len(self.sprites)
        self.left_img = pygame.transform.scale(pygame.image.load(self.sprites[self.current_sprite]), (120, 160))
        self.right_img = pygame.transform.flip(self.left_img, 1, 0)
        self.draw()
        
    def draw(self, new_location = False):
        if new_location:
           self.location = new_location
        self.location[1] = height - self.location[1]
        if self.facing_right:
            window.blit(self.right_img,self.location)
        else:
            window.blit(self.left_img,self.location)
        self.location[1] = height - self.location[1]
        
    def copy(self):
        temp = character(self.sprites, max_speed = self.max_speed, can_fly = self.can_fly, jump_speed = self.jump_speed, current_sprite = self.current_sprite)
        return temp

    def calc_dmg(self, other):
        if self.location[1] > 520:
            self.health -= 5
        if other.location[1] < self.location[1] < other.location[1] + 160:
            if other.location[0] - 80 < self.location[0] < other.location[0] + 80:
                other.health -= self.dmg
                
#character initialize
aoba = character(['aoba.png'])
yagami = character(['yagami.png'])
rin = character(['rin.png'])
hifumi = character(['hifumi.png'], max_speed = 20)
hajime = character(['hajime.png'])
yun = character(['yun.png'])
umiko = character(['umiko.png'])
nene = character(['nene.png'])
shizuku = character(['shizuku.png'])
momiji = character(['momiji.png'])
narumi = character(['narumi.png'])
hotaru = character(['hotaru.png'])
yamato = character(['yamato.png'])
sakura = character(['sakura.png'])
goku = character(['goku.png'], can_fly = True, jump_speed = 13)
saitama = character(['saitama.png'])
touru = character(['touru.png'], can_fly = True, jump_speed = 13)
kanna = character(['kanna.png', 'kanna2.png'], can_fly = True, jump_speed = 13, dmg = 3, max_speed = 18)
charlist = [aoba, yagami, rin, hifumi, hajime, yun, umiko, nene, shizuku, momiji, narumi, hotaru, yamato, sakura, goku, saitama, kanna, touru]


#character select
selecting_character = True
pos = 0
current_player = 1
playerchars = []
while selecting_character and playing:
    text("player " + str(current_player),[width/2,60])
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playing = False
        window.fill(backcolour)
        charlist[pos].draw([300,420])
        text(charlist[pos].name)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                pos = (pos + 1)%len(charlist)
            if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                pos = (pos - 1)%len(charlist)
            if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                charlist[pos].sprite_change()
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                charlist[pos].sprite_change(-1)
            if event.key == pygame.K_RETURN:
                selecting_character = 2 - current_player
                playerchars.append(charlist[pos].copy())
                current_player += 1

#gameplay
if playing:
    playerchars[0].draw([width * 0.25, 520])
    playerchars[1].draw([width * 0.65, 520])
    directions_pressed = [[0, 0],[0, 0]]
    
while playing:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playing = False
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            down = (event.type == pygame.KEYDOWN)
            for i in 0,1:
                for k in range(4):
                    directions_pressed[i][k > 1] += (2 * (k % 2) - 1) * (2 * down - 1) * (event.key == keys[i][k])                    
                       
    #window.blit(background,(0, 0))
    window.fill(backcolour)
    for i in 0,1:        
        playerchars[i].acceleration[0] = directions_pressed[i][0]
        if playerchars[i].location[1] == 200 or playerchars[i].can_fly:
            if directions_pressed[i][1] > 0:
                playerchars[i].speed[1] = playerchars[i].jump_speed
        playerchars[i].calc_dmg(playerchars[1-i])
    
    text(playerchars[0].name + " " + str(playerchars[0].health), [width * 1/3, 60], 60)
    text(playerchars[1].name + " " + str(playerchars[1].health), [width * 2/3, 60], 60)
    move_characters(playerchars)
    
    if playerchars[0].health < 1 or playerchars[1].health < 1:
        window.fill(backcolour)
        text("player " + str((playerchars[0].health < playerchars[1].health) + 1) + " wins" ,[width/2, height/2])
        pygame.display.update()
        while playing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False
                    
    pygame.display.update()
    clock.tick(60)
    
pygame.quit()