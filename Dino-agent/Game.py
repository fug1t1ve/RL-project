from utils import *
from init import *
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
class Game:
    def __init__(self,custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path = chrome_driver_path,chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        try:
            self._driver.get('chrome://dino')
        except WebDriverException:
            pass
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)
    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")
    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")
    def end(self):
        self._driver.close()

class Game_sate:
    def __init__(self,agent,game):
        self._agent = agent
        self._game = game
        self._display = show_img() 
        self._display.__next__() 
    def get_state(self,actions):
        actions_df.loc[len(actions_df)] = actions[1] 
        score = self._game.get_score() 
        reward = 0.1
        is_over = False 
        if actions[1] == 1:
            self._agent.jump()
        image = grab_screen(self._game._driver) 
        self._display.send(image) 
        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score 
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over