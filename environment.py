# coding:utf-8
"""
@author:lisheng
Created on 2021/12/15 10:41
"""

from game_sprite import *


class Game(object):
    def __init__(self):
        pygame.init()
        self.score = 0
        self.screen = pygame.display.set_mode(SCREEN_RECT.size)
        self.clock = pygame.time.Clock()
        self.__create_sprites()
        pygame.time.set_timer(CREATE_ENEMY_EVENT, 1000)
        pygame.time.set_timer(PLANE_FIRE_EVENT, 500)

    def __create_sprites(self):
        """创建精灵和精灵组"""

        # 创建背景精灵和精灵组
        bg = Background()
        bg1 = Background(True)
        self.back_group = pygame.sprite.Group(bg, bg1)
        # self.back_group = pygame.sprite.Group(bg)

        # 创建敌机精灵和精灵组
        self.enemy_group = pygame.sprite.Group()

        # 创建玩家精灵和玩家精灵组
        self.player = Player()
        self.player_group = pygame.sprite.Group(self.player)

        # 创建基线精灵和基线精灵组
        self.base_line = BaseLine()
        self.base_line_group = pygame.sprite.Group(self.base_line)

    def __event_handler(self):
        """事件检测"""
        for event in pygame.event.get():
            # 判断是否退出游戏
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == CREATE_ENEMY_EVENT:
                enemy = Enemy()
                # 将敌机精灵添加到敌机精灵组
                self.enemy_group.add(enemy)
            elif event.type == PLANE_FIRE_EVENT:
                self.player.fire()

    def __check_collide(self):
        # 1. 子弹摧毁敌机
        self.is_hit = pygame.sprite.groupcollide(self.player.bullets, self.enemy_group, True, True)
        if self.is_hit:
            self.score += 1
        # 2. 敌机撞毁玩家
        self.destroy = pygame.sprite.spritecollide(self.player, self.enemy_group, True)
        # 3. 敌机撞到基线
        self.is_endline = pygame.sprite.groupcollide(self.base_line_group, self.enemy_group, False, False)

    def __update_sprites(self):
        """更新精灵组"""
        self.back_group.update()
        self.back_group.draw(self.screen)

        self.enemy_group.update()
        self.enemy_group.draw(self.screen)

        self.player_group.update()
        self.player_group.draw(self.screen)

        self.player.bullets.update()
        self.player.bullets.draw(self.screen)

        self.base_line_group.draw(self.screen)

    def __show_score(self):
        score_font = pygame.font.Font(None, 36)
        score_text = score_font.render("Score: " + str(self.score), True, (128, 128, 128))
        text_rect = score_text.get_rect()
        text_rect.topleft = [10, 10]
        self.screen.blit(score_text, text_rect)
        pygame.display.update()

    def __check_input(self, actions):
        if actions[0] == 1:
            self.player.speed -= 2
        if actions[1] == 1:
            self.player.speed += 2
        if actions[2] == 1:
            self.player.speed = 0

    def step(self, actions):
        rewrd = 0.1
        terminal = False
        self.is_hit = pygame.sprite.groupcollide(self.player.bullets, self.enemy_group, True, True)
        self.is_endline = pygame.sprite.groupcollide(self.base_line_group, self.enemy_group, False, False)
        # self.__show_score()
        self.__check_input(actions)
        self.clock.tick(FRAME_PER_SEC)
        self.__event_handler()
        self.__check_collide()
        if self.is_hit:
            rewrd = 2
        if self.is_endline:
            terminal = True
            self.__init__()
            rewrd = -2
        self.__update_sprites()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        return image_data, rewrd, terminal


if __name__ == '__main__':
    # 创建游戏对象
    game = Game()

    # game.start_game()
    while True:
        a = [0, 0, 1]
        game.step(a)
