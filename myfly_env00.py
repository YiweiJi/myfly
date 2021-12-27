import sys
import pygame
import random
from pathlib import Path
from pygame.locals import *

import matplotlib
import matplotlib.pyplot as plt

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FRAME_PER_SEC = 30  # 帧数
PLAYER_SPEED = 30
NOOB_SPEED = 20
ACE_SPEED = 30
NOOB_HP = 1
ACE_HP = 2
ENEMY_FREQUENCY = 10
ENEMY_ACE_FREQUENCY = 70
ACE_APPEAR_SCORE = 20

curr_path=Path.cwd()

bgIMG = pygame.image.load(curr_path / 'bg.png')
playerIMG = pygame.image.load(curr_path / 'player.png')
enemynoobIMG = pygame.image.load(curr_path / 'enemynoob.png')
enemyaceIMG = pygame.image.load(curr_path / 'enemyace.png')
bulletIMG = pygame.image.load(curr_path / 'bullet.png')
enemyburstIMG = pygame.image.load(curr_path / 'burst.png')

bgimg = pygame.transform.smoothscale(bgIMG, (SCREEN_WIDTH, SCREEN_HEIGHT))

overimg = pygame.transform.smoothscale(bgIMG, (SCREEN_WIDTH, SCREEN_HEIGHT))

playerimg = pygame.transform.smoothscale(playerIMG,
                                         (SCREEN_WIDTH  *playerIMG.get_rect().width // bgIMG.get_rect().width, \
                                          SCREEN_HEIGHT *playerIMG.get_rect().height // bgIMG.get_rect().height))

enemynoobimg = pygame.transform.smoothscale(enemynoobIMG,
                                            (SCREEN_WIDTH * enemynoobIMG.get_rect().width // bgIMG.get_rect().width, \
                                             SCREEN_HEIGHT * enemynoobIMG.get_rect().height // bgIMG.get_rect().height))

enemyaceimg = pygame.transform.smoothscale(enemyaceIMG,
                                           (SCREEN_WIDTH * enemyaceIMG.get_rect().width // bgIMG.get_rect().width, \
                                            SCREEN_HEIGHT * enemyaceIMG.get_rect().height // bgIMG.get_rect().height))

bulletimg = pygame.transform.smoothscale(bulletIMG,
                                         (SCREEN_WIDTH * bulletIMG.get_rect().width // bgIMG.get_rect().width, \
                                          SCREEN_HEIGHT * bulletIMG.get_rect().height // bgIMG.get_rect().height))

enemyburstimg = pygame.transform.smoothscale(enemyburstIMG,
                                             (SCREEN_WIDTH * enemynoobIMG.get_rect().width // bgIMG.get_rect().width, \
                                              SCREEN_HEIGHT * enemynoobIMG.get_rect().height // bgIMG.get_rect().height))

class Player(pygame.sprite.Sprite):
    def __init__(self, init_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = playerimg
        self.rect = self.image.get_rect()
        self.rect.midbottom = init_pos  # 初始化玩家的坐标
        self.speed = PLAYER_SPEED
        self.bullets_group = pygame.sprite.Group()
        self.is_over = False

    def moveLeft(self):

        if self.rect.left <= 0:
            self.rect.left = 0
        else:
            self.rect.left -= self.speed

    def moveRight(self):

        if self.rect.left >= SCREEN_WIDTH - self.rect.width:
            self.rect.left = SCREEN_WIDTH - self.rect.width
        else:
            self.rect.left += self.speed

    def shoot(self):
        bullet = Bullet(self.rect.midtop)
        self.bullets_group.add(bullet)

    # def shoot(self,bulletimg,bullet_img_rect,score):
    #     bullet = Bullet(bulletimg,bullet_img_rect,self.rect.midtop)
    #     self.bullets.add(bullet)
    #     reward = -0.25
    #     score=score-0.3
    #     return score,reward


class Bullet(pygame.sprite.Sprite):
    def __init__(self, init_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = bulletimg
        self.rect = self.image.get_rect()
        self.rect.midbottom = init_pos
        self.speed = self.rect.height

    def move(self):
        self.rect.top -= self.speed


class Enemy(pygame.sprite.Sprite):
    def __init__(self, enemy_img, init_pos, speed, maxhp):
        pygame.sprite.Sprite.__init__(self)
        self.image = enemy_img
        self.enemyburstimg = enemyburstimg
        self.rect = self.image.get_rect()
        self.rect.midbottom = init_pos
        self.speed = speed
        self.speed2 = self.rect.height
        self.maxhp = maxhp

    def move(self):
        self.rect.top += self.speed



class GameState():
    def __init__(self):
        # 初始化 pygame
        pygame.init()
        # 设置游戏标题、图标，设置游戏界面大小
        pygame.display.set_caption('飞机游戏')
        icon = pygame.image.load(curr_path / 'ufo.png')
        pygame.display.set_icon(icon)

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.player = Player(self.screen.get_rect().midbottom)
        self.enemies_group = pygame.sprite.Group()  # 敌机精灵组
        self.enemies_down_group = pygame.sprite.Group()  # 敌机击毁精灵组

        self.shoot_frequency = 0  # 频率
        self.enemy_frequency = 0
        self.enemy_ace_frequency = 0
        self.score = 0
        self.reward = 0
        # self.reward1=0
        self.terminal = False
        self.rewards=0
        self.a  =0
        self.Reward=[]
        self.Action=[]
        self.cycle=0
        self.ax=[]
        self.ay=[]
        self.aveReward = 0
        self.count = 0
        #self.rewards += self.reward
    def frame_step(self, input_actions, chose_type):
        #print("indd",input_actions)
        self.reward = 0
        # self.reward1=0
        self.terminal = self.game_mode(input_actions, chose_type)

        self.off_screen_event(input_actions)
        self.create_enemy()
        self.screen.blit(bgimg, (0, 0))
        pygame.draw.line(self.screen, (255,255,255), (0,SCREEN_HEIGHT - self.player.rect.height*1.5), (SCREEN_WIDTH,SCREEN_HEIGHT - self.player.rect.height*1.5), 1)
        #self.screen.fill(0)
        self.screen.blit(playerimg, self.player.rect)

        self.collision_event(input_actions,chose_type)
        self.rewards += self.reward

        #print("terminal",self.terminal)
        if self.terminal:
            self.cycle += 1
            self.aveReward += self.rewards
        # 显示子弹和敌机
        self.player.bullets_group.draw(self.screen)
        self.enemies_group.draw(self.screen)
        # print('=========self.reward:', self.reward)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        if self.reward == -10:
            self.player = Player(self.screen.get_rect().midbottom)
            self.enemies_group = pygame.sprite.Group()  # 敌机精灵组
            self.enemies_down_group = pygame.sprite.Group()  # 敌机击毁精灵组
            if self.cycle % 20 == 0:
                self.aveReward = self.aveReward/20
                self.ax.append(self.cycle)
                self.ay.append(self.aveReward)
                self.aveReward = 0
            self.shoot_frequency = 0  # 频率
            self.enemy_frequency = 0
            self.enemy_ace_frequency = 0
            self.score = 0
            self.reward = 0
            self.count = 0
            self.terminal = False
            self.rewards = 0
            self.a=0
            self.Action=[]
            self.Reward=[]
            #self.input_action=0
        plt.ion()  # 开启一个画图的窗口
        plt.clf()  # 清除之前画的图
        plt.plot(self.ax, self.ay)  # 画出当前 ax 列表和 ay 列表中的值的图形
        plt.pause(0.01)
        plt.show()  # 关闭画图的窗口
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        clock = pygame.time.Clock()
        clock.tick(FRAME_PER_SEC)

        return image_data, self.reward, self.terminal

    def create_enemy(self):
        if self.enemy_frequency % ENEMY_FREQUENCY == 0 and self.count < 5:
            # enemy_noob_pos = [random.randint(0, SCREEN_WIDTH - enemynoobimg.get_rect().width), random.randint(0, SCREEN_HEIGHT//4)]
            enemy_noob_pos = [random.randint(0, SCREEN_WIDTH - enemynoobimg.get_rect().width), 10]
            enemy_noob = Enemy(enemynoobimg, enemy_noob_pos, NOOB_SPEED, NOOB_HP)
            self.enemies_group.add(enemy_noob)
            self.count += 1
        self.enemy_frequency += 1
        if self.enemy_frequency >= ENEMY_FREQUENCY and self.count == 5:
            # self.a = random.randint(0, SCREEN_WIDTH - enemynoobimg.get_rect().width)
            self.enemy_frequency = 0
        if self.enemy_frequency % ENEMY_FREQUENCY == 9 and self.count == 5:
            enemy_noob_pos = [10, 10]
            enemy_noob = Enemy(enemynoobimg, enemy_noob_pos, NOOB_SPEED, NOOB_HP)
            self.enemies_group.add(enemy_noob)
            self.enemy_frequency = 0
            self.count += 1
        if self.enemy_frequency % 4 == 3 and self.count == 6:
            enemy_noob_pos = [700, 10]
            enemy_noob = Enemy(enemynoobimg, enemy_noob_pos, NOOB_SPEED, NOOB_HP)
            self.enemies_group.add(enemy_noob)
            self.enemy_frequency = 0
            self.count += 1
        if self.enemy_frequency % 4 == 3 and self.count == 7:
            enemy_noob_pos = [700, 10]
            enemy_noob = Enemy(enemynoobimg, enemy_noob_pos, NOOB_SPEED, NOOB_HP)
            self.enemies_group.add(enemy_noob)
            self.enemy_frequency = 1
            self.count += 1
        if self.count >= 8:
            self.count = 0

            """
            # 生成敌机hp=2
            if self.enemy_ace_frequency % ENEMY_ACE_FREQUENCY == 0 and self.score > ACE_APPEAR_SCORE:
                #enemy_ace_pos = [0, 0]
                enemy_ace_pos = [random.randint(0, SCREEN_WIDTH - enemynoobimg.get_rect().width), random.randint(0, SCREEN_HEIGHT//4)]
                enemy_ace = Enemy(enemyaceimg, enemy_ace_pos, ACE_SPEED, ACE_HP)
                self.enemies_group.add(enemy_ace)
            self.enemy_ace_frequency += 1
            if self.enemy_ace_frequency >= ENEMY_ACE_FREQUENCY:
                self.enemy_ace_frequency = 0
            """
    def collision_event(self,input_actions,chose_type):

        # 判断是否击中敌机，击中hp-1,若hp=0，加入其他精灵组中
        enemies_if_hitted = pygame.sprite.groupcollide(self.enemies_group, self.player.bullets_group,True,True)
        for enemy_if_hitted in enemies_if_hitted:
            enemy_if_hitted.maxhp -= 1
            if enemy_if_hitted.maxhp == 0:
                self.enemies_group.remove(enemy_if_hitted)
                self.screen.blit(enemy_if_hitted.enemyburstimg, enemy_if_hitted.rect)
                self.enemies_down_group.add(enemy_if_hitted)
        for enemy_down in self.enemies_down_group:
            self.score += 1
            self.reward = 1
            self.enemies_down_group.remove(enemy_down)
        # 计算分数
        if input_actions[0]==1:
            input_actions='L'
        elif input_actions[1]==1:
            input_actions='R'
        elif input_actions[2]==1:
            input_actions='S'
        if len(list(self.Action))==20 and len(self.Reward)==20:
            self.Action = self.Action[1:]
            self.Action.append(input_actions)
            self.Reward = self.Reward[1:]
            self.Reward.append(self.reward)
        else:
            self.Action.append(input_actions)
            self.Reward.append(self.reward)
        if chose_type=="auto":
            Action_font = pygame.font.Font(None, SCREEN_WIDTH // 31)
            Action_text = f'A:{list(self.Action)}'
            Action_render = Action_font.render(Action_text, True, (128, 128, 128))
            self.screen.blit(Action_render, (4, 90))
            Reward_font = pygame.font.Font(None, SCREEN_WIDTH // 31)
            Reward_text = f'R:{self.Reward}'
            Reward_render = Reward_font.render(Reward_text, True, (128, 128, 128))
            self.screen.blit(Reward_render, (4, 110))
            cycle_font = pygame.font.Font(None, SCREEN_WIDTH // 31)
            cycle_text = f'Cycle:{self.cycle}'
            cycle_render = cycle_font.render(cycle_text, True, (128, 128, 128))
            self.screen.blit(cycle_render, (4, 70))
            # 得分
            score_font = pygame.font.Font(None, SCREEN_WIDTH // 31)
            score_text = f'score:{self.score}'
            score_render = score_font.render(score_text, True, (128, 128, 128))
            self.screen.blit(score_render, (4, 50))
            reward_font = pygame.font.Font(None, SCREEN_WIDTH // 31)
            reward_text = f'total_reward:{self.rewards}'
            reward_render = reward_font.render(reward_text, True, (128, 128, 128))
            self.screen.blit(reward_render, (4, 30))
        elif chose_type=="play" or "manual":
            cycle_font = pygame.font.Font(None, SCREEN_WIDTH // 31)
            cycle_text = f'Cycle:{self.cycle}'
            cycle_render = cycle_font.render(cycle_text, True, (128, 128, 128))
            self.screen.blit(cycle_render, (4, 50))
        # 得分
            score_font = pygame.font.Font(None, SCREEN_WIDTH // 31)
            score_text = f'score:{self.score}'
            score_render = score_font.render(score_text, True, (128, 128, 128))
            self.screen.blit(score_render, (4, 30))

        #return self.Action,self.Reward

    def off_screen_event(self,input_actions):
        # 移动子弹，移出屏幕后删除子弹
        for bullet in self.player.bullets_group:

            bullet.move()
            if bullet.rect.bottom <= 0:
                self.player.bullets_group.remove(bullet)

        # 移动敌机，检测是否游戏结束
        for enemy in self.enemies_group:
            enemy.move()
        for enemy in self.enemies_group:
            # if pygame.sprite.spritecollideany(enemy,self.player.bullets_group):
                # self.reward1=1
            if enemy.rect.top >= SCREEN_HEIGHT:
                self.a += 1
                self.enemies_group.remove(enemy)
            #print('self.player.rect.top - enemy.rect.bottom:', self.player.rect.top - enemy.rect.bottom)
            #print('self.player.rect.height * 1.5:', self.player.rect.height * 1.5)
            # if self.player.rect.top - enemy.rect.bottom > self.player.rect.height * 0.5:
            #     if self.player.rect.centerx < enemy.rect.centerx :
            #         if input_actions[1] == 1:
            #             self.reward = 0.1
            #         else:
            #             self.reward = -0.1
            #     # print("enemy.rect.right",self.player.rect.height)
            #         break
            #     if self.player.rect.centerx > enemy.rect.centerx :
            #         if input_actions[0] == 1:
            #             self.reward = 0.1
            #         else:
            #             self.reward = -0.1
            #         break

            if self.player.rect.top - enemy.rect.bottom <= self.player.rect.height * 0.5:
                if pygame.sprite.collide_rect(self.player,enemy) or self.a >= 2:
                    self.enemies_group.remove(enemy)
                    self.player.is_over = True
                    print('-------end of round------ score:', self.score)
                    # self.reward = 0
                    self.reward = -10
                    self.terminal = True
                    break
                # if self.player.rect.centerx < enemy.rect.centerx:
                #     if input_actions[0] == 1:
                #         self.reward = 0.1
                #     else:
                #         self.reward = -0.1
                #     break
                # if self.player.rect.centerx > enemy.rect.centerx:
                #     if input_actions[1] == 1:
                #         self.reward = 0.1
                #     else:
                #         self.reward = -0.1
                #     break



    def game_mode(self, input_actions, game_type):
        terminal = False
        key_pressed = pygame.key.get_pressed()
        if game_type == "manual":
            if key_pressed[K_a] or key_pressed[K_LEFT]:
                self.player.moveLeft()
            if key_pressed[K_d] or key_pressed[K_RIGHT]:
                self.player.moveRight()
            if key_pressed[K_SPACE]:
                if self.shoot_frequency % 2 == 0:
                    self.player.shoot()
                    if self.shoot_frequency >= 2:
                        self.shoot_frequency = 0
                self.shoot_frequency += 1

            if self.player.is_over:
                terminal = True
                while True:
                    score_font = pygame.font.Font(None, SCREEN_WIDTH // 10)
                    score_text = f'Score:{self.score}'
                    again_text = f'Again'
                    over_text = f'Exit'
                    again_x = SCREEN_WIDTH // 8
                    again_y = SCREEN_HEIGHT // 1.5
                    over_x = SCREEN_WIDTH // 1.5
                    over_y = SCREEN_HEIGHT // 1.5
                    score_render = score_font.render(score_text, True, (125, 125, 125))
                    again_render = score_font.render(again_text, True, (0, 0, 0))
                    over_render = score_font.render(over_text, True, (0, 0, 0))

                    if check_mouse(again_render.get_size(), again_x, again_y, pygame.mouse.get_pos()):
                        again_render = score_font.render(again_text, True, (70, 150, 150))

                    if check_mouse(over_render.get_size(), over_x, over_y, pygame.mouse.get_pos()):
                        over_render = score_font.render(over_text, True, (125, 125, 125))

                    self.screen.blit(bgimg, (0, 0))
                    self.screen.blit(score_render, (SCREEN_WIDTH // 3, SCREEN_HEIGHT // 4))
                    self.screen.blit(again_render, (again_x, again_y))
                    self.screen.blit(over_render, (over_x, over_y))

                    pygame.display.update()
                    mouse_down = pygame.mouse.get_pressed()
                    if mouse_down[0]:
                        pos = pygame.mouse.get_pos()
                        if again_x < pos[0] < again_x + again_render.get_width() and\
                                again_y < pos[1] < again_y + again_render.get_height():
                            main()
                            break
                        elif over_x < pos[0] < over_x + over_render.get_width() and \
                                over_y < pos[1] < over_y + over_render.get_height():
                            pygame.quit()
                            sys.exit()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()

        elif game_type == "play":
            if self.shoot_frequency % 4 == 0:
                self.player.shoot()
            if self.shoot_frequency >= 4:
                self.shoot_frequency = 0
            self.shoot_frequency += 1
            for enemy in self.enemies_group:
                if self.player.rect.top - enemy.rect.bottom > self.player.rect.height * 0.5:
                    if self.player.rect.centerx < enemy.rect.centerx:
                        self.player.moveRight()
                        # print("enemy.rect.right",self.player.rect.height)
                        break
                    if self.player.rect.centerx > enemy.rect.centerx:
                        self.player.moveLeft()
                        break

                if self.player.rect.top - enemy.rect.bottom <= self.player.rect.height * 0.5:
                    if self.player.rect.centerx < enemy.rect.centerx:
                        self.player.moveLeft()
                        break
                    if self.player.rect.centerx > enemy.rect.centerx:
                        self.player.moveRight()
                        break

        elif game_type == "auto":
            #or input_actions[3] == 1:
            if input_actions[0] == 1 or input_actions[1] == 1 or input_actions[2]==1:  # 检查输入
                if input_actions[0] == 1:
                    self.player.moveLeft()
                elif input_actions[1] == 1:
                    self.player.moveRight()
                else:
                    pass
                # elif input_actions[0] == 1:
                #     self.score,reward=self.player.shoot(self.bulletimg,self.bulletimg.get_rect(),self.score)
            else:
                raise ValueError('Multiple input actions!')
            #  判断游戏是否结束，如果未结束，子弹以一定频率发射
            # if not self.player.is_over:
            if self.shoot_frequency % 5 == 0:
                self.player.shoot()
            self.shoot_frequency += 1
            if self.shoot_frequency >= 5:
                self.shoot_frequency = 0
        return terminal


"""
def my_spritecollide(sprite, group_bullet):
    crashed = []
    append = crashed.append
    for bullet_sprite in group_bullet.sprites():
        enemy_x, enemy_y = sprite.rect.bottomleft
        enemy_centery = sprite.rect.centery
        bullet_x, bullet_y = bullet_sprite.rect.topleft
        if (bullet_y + (
                2 * bullet_sprite.speed) > enemy_centery and bullet_y < enemy_y):  # 判断两帧之间子弹沿Y轴方向经过的线段是否与敌机的Y坐标有过交点
            if (  # 飞机沿X正方向经过的线段是否与子弹坐标有过交点
                    sprite.speed > 0 and enemy_x - sprite.speed < bullet_x and bullet_x < enemy_x):
                bullet_sprite.kill()
                append(bullet_sprite)
            elif (sprite.speed < 0 and enemy_x - sprite.speed > bullet_x and bullet_x > enemy_x):  # 负方向
                bullet_sprite.kill()
                append(bullet_sprite)
    return crashed


def my_groupcollide(group_enemy, group_bullet):
    crashed = {}
    sprite_collide_func = my_spritecollide
    for enemy_sprite in group_enemy.sprites():
        collision = sprite_collide_func(enemy_sprite, group_bullet)  # 判断两帧之间，敌机和子弹经过的线段是否有过交点
        if collision:
            crashed[enemy_sprite] = collision
    return crashed

"""
def check_mouse(size, x, y, position):
    width, height = size
    x_match = x < position[0] < x + width
    y_match = y < position[1] < y + height

    if x_match and y_match:
        return True
    else:
        return False
