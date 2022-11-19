from nuscenes.prediction.input_representation.agents import *
import abc
from typing import List
import numpy as np

# Modified from CoverNet input representation here: https://github.com/nutonomy/nuscenes-devkit.

class StaticLayerRepresentation(abc.ABC):
    """ Represents static map information as a numpy array. """

    @abc.abstractmethod
    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        raise NotImplementedError()


class AgentRepresentation(abc.ABC):
    """ Represents information of agents in scene as numpy array. """

    @abc.abstractmethod
    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        raise NotImplementedError()


class Combinator(abc.ABC):
    """ Combines the StaticLayer and Agent representations into a single one. """

    @abc.abstractmethod
    def combine(self, data: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()


class InputRepresentationDecode:
    """
    Specifies how to represent the input for a prediction model.
    Need to provide a StaticLayerRepresentation - how the map is represented,
    an AgentRepresentation - how agents in the scene are represented,
    and a Combinator, how the StaticLayerRepresentation and AgentRepresentation should be combined.
    """

    def __init__(self, static_layer: StaticLayerRepresentation, agent: AgentRepresentation,
                 combinator: Combinator):

        self.static_layer_rasterizer = static_layer
        self.agent_rasterizer = agent
        self.combinator = combinator

    def make_input_representation(self, instance_token: str, sample_token: str) -> np.ndarray:

        static_layers = self.static_layer_rasterizer.make_representation(instance_token, sample_token)
        center_agent, social_context = self.agent_rasterizer.make_representation(instance_token, sample_token)

        return center_agent, static_layers, social_context, self.combinator.combine([static_layers, social_context, center_agent])

def draw_agent_boxes_separate(center_agent_annotation: Dict[str, Any],
                     center_agent_pixels: Tuple[float, float],
                     agent_history: History,
                     base_image: np.ndarray,
                     get_color: Callable[[str], Tuple[int, int, int]],
                     resolution: float = 0.1,
                     center_agent=True) -> None:
    """
    Draws past sequence of agent boxes on the image.
    :param center_agent_annotation: Annotation record for the agent
        that is in the center of the image.
    :param center_agent_pixels: Pixel location of the agent in the
        center of the image.
    :param agent_history: History for all agents in the scene.
    :param base_image: Image to draw the agents in.
    :param get_color: Mapping from category_name to RGB tuple.
    :param resolution: Size of the image in pixels / meter.
    :return: None.
    """

    agent_x, agent_y = center_agent_annotation['translation'][:2]

    for instance_token, annotations in agent_history.items():

        num_points = len(annotations)

        for i, annotation in enumerate(annotations):
            
            if center_agent and (instance_token == center_agent_annotation['instance_token']):

                box = get_track_box(annotation, (agent_x, agent_y), center_agent_pixels, resolution)

                color = (255, 0, 0)
                
                # Don't fade the colors if there is no history
                if num_points > 1:
                    color = fade_color(color, i, num_points - 1)

                cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)

            elif not center_agent and (instance_token != center_agent_annotation['instance_token']):
                box = get_track_box(annotation, (agent_x, agent_y), center_agent_pixels, resolution)
                color = get_color(annotation['category_name'])

                # Don't fade the colors if there is no history
                if num_points > 1:
                    color = fade_color(color, i, num_points - 1)

                cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)

class AgentBoxesWithFadedHistorySeparate(AgentRepresentation):
    """
    Represents the past sequence of agent states as a three-channel
    image with faded 2d boxes.
    """

    def __init__(self, helper: PredictHelper,
                 seconds_of_history: float = 2,
                 frequency_in_hz: float = 2,
                 resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25,
                 color_mapping: Callable[[str], Tuple[int, int, int]] = None):

        self.helper = helper
        self.seconds_of_history = seconds_of_history
        self.frequency_in_hz = frequency_in_hz

        if not resolution > 0:
            raise ValueError(f"Resolution must be positive. Received {resolution}.")

        self.resolution = resolution

        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right

        if not color_mapping:
            color_mapping = default_colors

        self.color_mapping = color_mapping

    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        """
        Draws agent boxes with faded history into a black background.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :return: np.ndarray representing a 3 channel image.
        """

        # Taking radius around track before to ensure all actors are in image
        buffer = max([self.meters_ahead, self.meters_behind,
                      self.meters_left, self.meters_right]) * 2

        image_side_length = int(buffer/self.resolution)

        # We will center the track in the image
        central_track_pixels = (image_side_length / 2, image_side_length / 2)

        base_image_center_agent = np.zeros((image_side_length, image_side_length, 3))
        base_image_social_context = np.zeros((image_side_length, image_side_length, 3))

        history = self.helper.get_past_for_sample(sample_token,
                                                  self.seconds_of_history,
                                                  in_agent_frame=False,
                                                  just_xy=False)
        history = reverse_history(history)

        present_time = self.helper.get_annotations_for_sample(sample_token)

        history = add_present_time_to_history(present_time, history)

        center_agent_annotation = self.helper.get_sample_annotation(instance_token, sample_token)

        draw_agent_boxes_separate(center_agent_annotation, central_track_pixels,
                         history, base_image_center_agent, resolution=self.resolution, get_color=self.color_mapping, center_agent=True)
        
        draw_agent_boxes_separate(center_agent_annotation, central_track_pixels,
                         history, base_image_social_context, resolution=self.resolution, get_color=self.color_mapping, center_agent=False)

        center_agent_yaw = quaternion_yaw(Quaternion(center_agent_annotation['rotation']))
        rotation_mat = get_rotation_matrix(base_image_center_agent.shape, center_agent_yaw)

        rotated_image_center_agent = cv2.warpAffine(base_image_center_agent, rotation_mat, (base_image_center_agent.shape[1],
                                                                  base_image_center_agent.shape[0]))

        rotated_image_social_context = cv2.warpAffine(base_image_social_context, rotation_mat, (base_image_social_context.shape[1],
                                                                  base_image_social_context.shape[0]))        
        
        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind,
                                       self.meters_left, self.meters_right, self.resolution,
                                       image_side_length)

        return rotated_image_center_agent[row_crop, col_crop].astype('uint8'), rotated_image_social_context[row_crop, col_crop].astype('uint8')