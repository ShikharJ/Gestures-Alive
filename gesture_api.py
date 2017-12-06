import numpy
import math


class Gesture(object):

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_palm(self, hand_center, hand_radius):
        self.hand_center = hand_center
        self.hand_radius = hand_radius

    def set_finger_position(self, finger_position):
        self.finger_position = finger_position
        self.finger_count = len(finger_position)

    def calculate_angles(self):
        self.angle = numpy.zeros(self.finger_count, dtype=int)
        for i in range(self.finger_count):
            x = self.finger_position[i][0]
            y = self.finger_position[i][1]
            self.angle[i] = abs(math.atan2((self.hand_center[1] - y), (x - self.hand_center[0])) * 180 / math.pi)


def define_gestures():

    dictionary = {}

    v = Gesture("V Sign")
    v.set_palm((475, 225), 45)
    v.set_finger_position([(490, 90), (415, 105)])
    v.calculate_angles()
    dictionary[v.get_name()] = v

    l_right = Gesture("L Sign")
    l_right.set_palm((475, 225), 50)
    l_right.set_finger_position([(450, 62), (345, 200)])
    l_right.calculate_angles()
    dictionary[l_right.get_name()] = l_right

    index_pointing = Gesture("Index Pointing")
    index_pointing.set_palm((480, 230), 43)
    index_pointing.set_finger_position([(475, 102)])
    index_pointing.calculate_angles()
    dictionary[index_pointing.get_name()] = index_pointing

    return dictionary


def compare_gestures(primary, secondary):

    if primary.finger_count == secondary.finger_count:
        if primary.finger_count == 1:
            angle_difference = primary.angle[0] - secondary.angle[0]
            if angle_difference > 20:
                result = 0
            else:
                primary_length = numpy.sqrt((primary.finger_position[0][0] - primary.hand_center[0]) ** 2
                                            + (primary.finger_position[0][1] - primary.hand_center[1]) ** 2)
                secondary_length = numpy.sqrt((secondary.finger_position[0][0] - secondary.hand_center[0]) ** 2
                                              + (secondary.finger_position[0][1] - secondary.hand_center[1]) ** 2)
                length_difference = primary_length / secondary_length
                radius_difference = primary.hand_radius / secondary.hand_radius
                length_score = abs(length_difference - radius_difference)
                if length_score < 0.09:
                    result = secondary.get_name()
                else:
                    result = 0
        else:
            angle_difference = []
            for i in range(primary.finger_count):
                angle_difference.append(primary.angle[i] - secondary.angle[i])
            angle_score = max(angle_difference) - min(angle_difference)
            if angle_score < 15:
                length_difference = []
                for i in range(primary.finger_count):
                    primary_length = numpy.sqrt((primary.finger_position[i][0] - primary.hand_center[0]) ** 2 + (primary.finger_position[i][1] - primary.hand_center[1]) ** 2)
                    secondary_length = numpy.sqrt((secondary.finger_position[i][0] - secondary.hand_center[0]) ** 2 + (secondary.finger_position[i][1] - secondary.hand_center[1]) ** 2)
                    length_difference.append(primary_length / secondary_length)
                length_score = max(length_difference) - min(length_difference)
                if length_score < 0.06:
                    result = secondary.get_name()
                else:
                    result = 0
            else:
                result = 0
    else:
        result = 0

    return result


def decide_gesture(source, gesture_dictionary):

    for k in gesture_dictionary.keys():
        result = compare_gestures(source, gesture_dictionary[k])

        if result != 0:
            return result

    return "None"
