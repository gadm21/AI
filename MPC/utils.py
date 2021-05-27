
import numpy as np
import time
import csv
import numpy as np
from PIL import Image
import cv2
from numpy.lib.twodim_base import triu_indices_from
from scipy.optimize import minimize, differential_evolution
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime



class CUtils(object):
    def __init__(self):
        pass

    def create_var(self, var_name, value):
        if not var_name in self.__dict__:
            self.__dict__[var_name] = value



class State:
    def __init__(self, path = None, X=0, Y=0, th=0, V=0, deg = True):
        self.x = X
        self.y = Y
        self.v = V
        self.cte = 0
        self.eth = 0
        if deg :self.th = self.deg2rad(th)
        else: self.th = th

        if path is not None: self.cte, self.eth = self.calculate_error(path)


    def deg2rad(self, deg):
        rad = deg * np.pi / 180
        return rad
    
    def rad2deg(self, rad):
        deg = rad * 180 / np.pi
        return deg

    def calculate_error(self, path):
        coeff = np.polyfit(path[:, 0], path[:, 1], 2)
        th_des = np.arctan(coeff[1] + 2*coeff[0]*self.x)
        eth = self.th - th_des
        cte = np.polyval(coeff,self.x) - self.y
        return cte, eth

    def __str__(self):
        return "x:{:.2f}  y:{:.2f}   th(deg):{:.2f}   v:{:.2f}    cte:{:.2f}   eth:{:.2f}"\
            .format(self.x, self.y, self.rad2deg(self.th), self.v, self.cte, self.eth) 




class inputs:
    def __init__(self, steer_angle=0, accelartion=0):
        self.steer_angle = steer_angle
        self.accelartion = accelartion


    
def rad2deg( rad):
    deg = rad * 180 / np.pi
    return deg

def deg2rad(deg):
    rad = deg * np.pi / 180
    return rad



def model(inputs, init_state, coff, dt = 0.1, L = 3):


    def deg2rad(deg):
        rad = deg * np.pi / 180
        return rad

    final_state = State()

    ## find the final satte after dt of time ###
    final_state.x  = init_state.x  + init_state.v*np.cos(init_state.th)*dt
    final_state.y  = init_state.y  + init_state.v*np.sin(init_state.th)*dt
    final_state.v  = init_state.v  + inputs.accelartion*dt

    final_state.th = init_state.th + (init_state.v/L)*deg2rad(inputs.steer_angle)*dt
    th_des = np.arctan(coff[1] + 2*coff[0]*init_state.x)
    final_state.eth = final_state.th - th_des

    final_state.cte = np.polyval(coff,final_state.x) - final_state.y

    return final_state




class Controller2D(object):
    def __init__(self, waypoints, initial_state, verbose = False):
        self.verbose = verbose
        self.vars                = CUtils()
        self.waypoints          = waypoints
        self.state = initial_state
        self.add_delay = False

        self.desired_speed      = 10

        self.timestamp  = 0
        self.throttle       = 0
        self.steer          = 0

        
    def deg2rad(self, deg):
        rad = deg * np.pi / 180
        return rad

    def __str__(self):
        
        return "Controller state :: " + str(self.state) + "  " + \
            "throttle:{:.2f}  steer(deg):{:.2f}  timestamp:{:.2f}".format(self.throttle, self.steer, self.timestamp)

    def update_values(self, new_state):
        self.state = new_state
        

    def get_coeffs(self):

        def find_closest_idx():
            value = self.state.x
            closest_idx = np.abs(self.waypoints[:, 0] - value).argmin()
            return closest_idx
        

        margin = 50
        idx = find_closest_idx()
        coeffs = np.polyfit(self.waypoints[idx-margin : idx+margin, 0], self.waypoints[idx-margin : idx+margin, 1], 2)
        return coeffs


    def update_controls(self):
        
        
        ## Define STEP TIME ##
        STEP_TIME = 0.1
        ## prediction horizen ##
        P = 10
        acc_offest = P
        ## init geuss ##
        x0 = np.zeros(2*P)

        ## cost function weights ##
        cte_W = 500
        eth_W = 100
        v_W = 100
        st_rate_W = 2
        acc_rate_W = 1
        st_W = 1
        acc_W = 1
        dir_W = 0


        ## cost function parameters normalization factors to approximately make each unit equal in "cost" to the other.
        normalize_distance = 0.05
        normalize_rad = 0.14
        normalize_speed = 0.5
        normalize_acc = 0.5

        ## input bounds ##
        b1 = (-30, 30)
        b2 = (-5.0, 5.0)
        bnds = tuple([b1]*P+[b2]*P)
        

        ## find COFF of the polynomial ##
        margin = 50
        # coff = np.polyfit(self.waypoints[int(self.state.x) - margin : int(self.state.x) + margin, 0], self.waypoints[int(self.state.x):, 1], 2)
        coff = self.get_coeffs()
        # print("coff:", coff)
        
        


        ######################################################
        ############ MODEL PREDCIIVE CONTROLLER ##############
        ######################################################
        
        # cost fun or objective fun req to minimaize ##
        def objective(x):
            u = inputs()
            Error = 0
            init_state_1 = self.state
            for i in range(P):
                u.steer_angle = x[i] 
                u.accelartion = x[i+acc_offest]
                next_state = model(u,init_state_1, coff, dt=STEP_TIME, L=3)
                # print("cteW:{}  ethW:{}    vW:{}".format(cte_W, eth_W, v_W))
                if True :
                    Error += cte_W*np.absolute(next_state.cte)*normalize_distance + \
                             eth_W*np.absolute(next_state.eth)*normalize_rad**2 + \
                             v_W*np.absolute(next_state.v - self.desired_speed)*normalize_speed**2 + \
                             st_W*np.absolute(deg2rad(u.steer_angle))*normalize_rad**2 + \
                             acc_W*np.absolute(u.accelartion)*normalize_acc**2 + \
                             dir_W*(init_state_1.x - next_state.x)*normalize_distance
                else:
                    Error += cte_W*np.absolute(next_state.cte) + eth_W*np.absolute(next_state.eth) + v_W*np.absolute(next_state.v - self.desired_speed) \
                            + st_rate_W*np.absolute(u.steer_angle - x[i-1]) + acc_rate_W*np.absolute(u.accelartion - x[i+acc_offest-1]) \
                            + st_W*np.absolute(u.steer_angle) + acc_W*np.absolute(u.accelartion) + dir_W*(init_state_1.x - next_state.x)
                init_state_1 = next_state
            return Error


        latency = 0.033 # 100 ms
        if self.add_delay:
            # # Let's predict the state. Rembember that px, py and psi wrt car are all 0.
            self.state.x = self.state.v * latency
            self.state.y = 0
            self.state.th = -self.state.v * self.steer * latency / 3
            self.state.v = self.state.v + (self.throttle)/ STEP_TIME * latency
            self.state.cte = self.state.cte + self.state.v * np.sin(self.state.eth) * latency
            self.state.eth = self.state.eth + self.state.th

        solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
        u = solution.x
        # print("solution:", u)
        self.steer = u[0]
        self.throttle = u[acc_offest]

        self.timestamp += STEP_TIME
        new_state = model(inputs(self.steer, self.throttle), self.state, coff)
        self.update_values(new_state)



'''

class Controller2D(object):
    def __init__(self, waypoints, verbose = False):
        self.verbose = verbose
        self.vars                = CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def map2Car(self, x, y, yaw, waypoints):  

        waypoints = np.array(waypoints)
        shift_x = waypoints[0] - x
        shift_y = waypoints[1] - y

        car_x = shift_x * np.cos(-yaw) - shift_y * np.sin(-yaw)
        car_y = shift_x * np.sin(-yaw) + shift_y * np.cos(-yaw)    

        return car_x, car_y 

    def map_coord_2_Car_coord(self, x, y, yaw, waypoints):  
    
        wps = np.squeeze(waypoints)
        wps_x = wps[:,0]
        wps_y = wps[:,1]

        num_wp = wps.shape[0]
        
        ## create the Matrix with 3 vectors for the waypoint x and y coordinates w.r.t. car 
        wp_vehRef = np.zeros(shape=(3, num_wp))
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
                

        wp_vehRef[0,:] = cos_yaw * (wps_x - x) - sin_yaw * (wps_y - y)
        wp_vehRef[1,:] = sin_yaw * (wps_x - x) + cos_yaw * (wps_y - y)        

        return wp_vehRef   

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0


        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.
            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)
            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0
            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('x_previous', 0.0)
        self.vars.create_var('y_previous', 0.0)
        self.vars.create_var('th_previous', 0.0)
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('cte_previous', 0.0)
        self.vars.create_var('eth_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)
        
        ## Define STEP TIME ##
        STEP_TIME = t - self.vars.t_previous
        ## prediction horizen ##
        P = 10
        acc_offest = P
        ## init geuss ##
        x0 = np.zeros(2*P)
        ## cost function weights ##
        cte_W = 50
        eth_W = 100
        v_W = 100
        st_rate_W = 200
        acc_rate_W = 10
        st_W = 100
        acc_W = 1
        ## input bounds ##
        b1 = (-1.22, 1.22)
        b2 = (0.0, 1.0)
        bnds = (b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b2,b2,b2,b2,b2,b2,b2,b2,b2,b2)
        # bnds = (b1,b1,b2,b2)
        wps_vehRef = self.map_coord_2_Car_coord(x, y, yaw, waypoints)
        wps_vehRef_x = wps_vehRef[0,:]
        wps_vehRef_y = wps_vehRef[1,:]

        ## find COFF of the polynomial ##
        coff = np.polyfit(wps_vehRef_x, wps_vehRef_y, 3)
        v_ref = v_desired
        # Skip the first frame to store previous values properly

        if self._start_control_loop:
            """
                Controller iteration code block.
                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position
                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            ############ MODEL PREDCIIVE CONTROLLER ##############
            ######################################################
            ######################################################
            # cost fun or objective fun req to minimaize ##
            def objective(x):
                u = inputs()
                Error = 0
                global init_state
                init_state_1 = init_state
                for i in range(P):
                    u.steer_angle = x[i-1] 
                    u.accelartion = x[i+acc_offest]
                    next_state = model(u,init_state_1, coff, dt=STEP_TIME, L=3)
                    if i == 0 :
                        Error += cte_W*np.absolute(next_state.cte)**2 + eth_W*np.absolute(next_state.eth)**2 + v_W*np.absolute(next_state.v - v_ref)**2 \
                                + st_W*np.absolute(u.steer_angle)**2 + acc_W*np.absolute(u.accelartion)**2
                    else:
                        Error += cte_W*np.absolute(next_state.cte)**2 + eth_W*np.absolute(next_state.eth)**2 + v_W*np.absolute(next_state.v - v_ref)**2 \
                                + st_rate_W*np.absolute(u.steer_angle - x[i-1])**2 + acc_rate_W*np.absolute(u.accelartion - x[i+acc_offest-1])**2 \
                                + st_W*np.absolute(u.steer_angle)**2 + acc_W*np.absolute(u.accelartion)**2
                    init_state_1 = next_state
                return Error

            CarRef_x = CarRef_y = CarRef_yaw = 0.0

            cte = np.polyval(coff, CarRef_x) - CarRef_y

            # get orientation error from fit ( Since we are trying a 3rd order poly, then, f' = a1 + 2*a2*x + 3*a3*x2)
            # in this case and since we moved our reference sys to the Car, x = 0 and also yaw = 0
            yaw_err = CarRef_yaw - np.arctan(coff[1])

            # I can send the ACTUAL state to the MPC or I can try to compensate for the latency by "predicting" what would 
            # be the state after the latency period.
            latency = 0.033 # 100 ms

            # # Let's predict the state. Rembember that px, py and psi wrt car are all 0.
            init_state.x = v * latency
            init_state.y = 0
            init_state.th = -v * self._set_steer * latency / 3
            init_state.v = v + (v - self.vars.v_previous)/ STEP_TIME * latency
            init_state.cte = cte + v * np.sin(yaw_err) * latency
            init_state.eth = yaw_err + init_state.th

            solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
            u = solution.x
            steer_output = u[0]

            if u[acc_offest] < 0 :
                brake_output = u[acc_offest]
            else:
                throttle_output = u[acc_offest]



            if self.verbose :
                print("[INFO] throttle_output: "+ str(throttle_output))
                print("[INFO] steer_output: "+ str(steer_output))
                # print("[INFO] X: "+ str(init_state.x))
                # print("[INFO] Y: "+ str(init_state.y))
                print("[INFO] TH: "+ str(init_state.th))
                print("[INFO] V: "+ str(init_state.v))
                print("[INFO] CTE: "+ str(init_state.cte))
                print("[INFO] ETH: "+ str(init_state.eth))
                # print("[INFO] COFF: "+ str(coff))
                print("______________________________________________")
                file = open('Errors.csv', 'a', newline='') 
                writer = csv.writer(file)
                writer.writerow([t, init_state.cte, init_state.eth])

   
            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.t_previous = t  # Store timestamp  to be used in next step
        self.vars.v_previous = v

'''






def show_image(image, name = "env", period = 60):
    cv2.imshow(name, image)
    if cv2.waitKey(period) & 0xFF == ord('q'):
        pass
    # cv2.destroyWindow("image")


def render(waypoints, controller, fig, show = True, arrow = True, clear_first = True, margin = 50, pause = 0.1):
    
    if not show : return 

    if clear_first : fig.clear()
    axes = plt.gca()
    axes.set_xlim([int(controller.state.x) - margin, int(controller.state.x + margin)])
    axes.set_ylim([int(controller.state.y) - margin, int(controller.state.y + margin)])

    plt.plot(waypoints[:, 0], waypoints[:, 1])
    plt.scatter(controller.state.x, controller.state.y, color = 'r', s = 4)
    if arrow :
        steering_angle = deg2rad(controller.steer + rad2deg(controller.state.th))
        from_point = (controller.state.x, controller.state.y )
        orientation_to_point = (from_point[0] + np.cos(controller.state.th)*10, from_point[1] + np.sin(controller.state.th)*10)
        steer_to_point = (from_point[0] + np.cos(steering_angle)*10, from_point[1] + np.sin(steering_angle)*10)

        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="k")
        plt.gca().add_patch(patches.FancyArrowPatch(from_point, orientation_to_point, **dict(arrowstyle=style, color="g")))
        plt.gca().add_patch(patches.FancyArrowPatch(from_point, steer_to_point, **dict(arrowstyle=style, color="r")))

        # print("from:", from_point)
        # print("orientation:", orientation_to_point)
        # # print("steering:", steer_to_point)

    plt.show(block = False)
    plt.pause(pause)