import pygame
pygame.init()
playing = True

pygame.display.set_caption('new game')
width = 1200
height = 720
black = (0,0,0)
white = (255,255,255)
backcolour = (0, 200 ,255)
window = pygame.display.set_mode((width,height))
window.fill(backcolour)
pygame.display.update()
clock = pygame.time.Clock()

def text(string, location = [800, 400], size = 115):
    str1 = pygame.font.Font('freesansbold.ttf',size).render(string, True, black)
    strec = str1.get_rect()
    strec.center = location
    window.blit(str1,strec)

class character:
    def __init__(self,img,location = [width * 0.45, height * 0.5],speed = 15):
        self.left_img = pygame.image.load(img)
        self.right_img = pygame.transform.flip(self.left_img,1,0)
        self.health = 100
        self.location = [width * 0.45, height * 0.5]
        self.facing_right = 0
        self.speed = speed
        self.name = img[:-4]
        
    def draw(self, new_location = False):
        if new_location:
           self.location = new_location 
        if self.facing_right:
            window.blit(self.right_img,self.location)
        else:
            window.blit(self.left_img,self.location)
            
    def move(self,location_change):
        self.location[0] = max(min(self.location[0] + location_change[0],width - 250), 0)
        self.location[1] = max(min(self.location[1] + location_change[1],height - 250), 100)
        self.draw()
        
    def copy(self):
        return character(self.name + ".png")

    def calc_dmg(self,other):
        if other.location[1] - 160 < self.location[1] < other.location[1]:
            if other.location[0] - 80 < self.location[0] < other.location[0] + 80:
                other.health -= 5
                
#character initialize
aoba = character('aoba.png')
yagami = character('yagami.png')
rin = character('rin.png')
hifumi = character('hifumi.png')
hajime = character('hajime.png')
yun = character('yun.png')
umiko = character('umiko.png')
nene = character('nene.png')
shizuku = character('shizuku.png')
momiji = character('momiji.png')
narumi = character('narumi.png')
hotaru = character('hotaru.png')
yamato = character('yamato.png')
charlist = [aoba, yagami, rin, hifumi, hajime, yun, umiko, nene, shizuku, momiji, narumi, hotaru, yamato]


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
        charlist[pos].draw([300,300])
        text(charlist[pos].name)
        pygame.display.update()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                pos = (pos + 1)%len(charlist)
            if event.key == pygame.K_DOWN:
                pos = (pos - 1)%len(charlist)
            if event.key == pygame.K_RETURN:
                selecting_character = 2 - current_player
                playerchars.append(charlist[pos].copy())
                current_player += 1

playerchars[0].draw([width * 0.25, height * 0.5])
playerchars[1].draw([width * 0.65, height * 0.5])
location_changes = [[0, 0],[0, 0]]
while playing:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playing = False
        
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            #player1
            movement = playerchars[0].speed if event.type == pygame.KEYDOWN else -playerchars[0].speed
            if event.key == pygame.K_LEFT:
                location_changes[0][0] -= movement
                if movement > 0:playerchars[0].facing_right = 0
            elif event.key == pygame.K_RIGHT:
                location_changes[0][0] += movement
                if movement > 0: playerchars[0].facing_right = 1
            if event.key == pygame.K_UP:
                location_changes[0][1] -= movement
            elif event.key == pygame.K_DOWN:
                location_changes[0][1] += movement
                                
            #player2
            movement = playerchars[1].speed if event.type == pygame.KEYDOWN else -playerchars[1].speed
            if event.key == pygame.K_a:
                location_changes[1][0] -= movement
                if movement > 0:playerchars[1].facing_right = 0
            elif event.key == pygame.K_d:
                location_changes[1][0] += movement
                if movement > 0: playerchars[1].facing_right = 1
            if event.key == pygame.K_w:
                location_changes[1][1] -= movement
            elif event.key == pygame.K_s:
                location_changes[1][1] += movement
     
    window.fill(backcolour)
    text(playerchars[0].name + " " + str(playerchars[0].health),[width /3,60],60)
    text(playerchars[1].name + " " + str(playerchars[1].health),[width*2/3,60],60)                 
    playerchars[0].move(location_changes[0])
    playerchars[1].move(location_changes[1])
    playerchars[0].calc_dmg(playerchars[1])
    playerchars[1].calc_dmg(playerchars[0])
    pygame.display.update()
    clock.tick(60)
    
pygame.quit()