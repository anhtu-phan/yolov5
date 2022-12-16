import torch
import os
import io
import json
import logging

from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


class ChickDetection(LabelStudioMLBase):

    def __init__(self, score_threshold=0.3, labels_file=None, **kwargs):
        super(ChickDetection, self).__init__(**kwargs)
        self.labels_file = labels_file
        self.score_threshold = score_threshold
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(self.parsed_label_config, 'RectangleLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name
        self.model = None
        print("========= DONE LOAD MODEL ========>>>>>>>>")

    def predict(self, tasks, **kwargs):
        self.model = torch.hub.load("ultralytics/yolov5", 'custom',
                                    path="/home/anhtu/projects/yolov5/checkpoint/best.pt")

        print("predict------>>>>>>>>", tasks)
        assert len(tasks) == 1
        results = []
        all_scores = []
        image_url = tasks[0]['data']['image']
        image_path = self.get_local_path(image_url)
        im = Image.open(image_path)
        img_width, img_height = get_image_size(image_path)
        pred = self.model([im], size=640)
        print("pred =========>>>>>>\n", pred.pandas().xyxy[0])
        for _, row in pred.pandas().xyxy[0].iterrows():
            output_label = self.label_map.get(row['name'], row['name'])
            if float(row['confidence']) > self.score_threshold and output_label in self.labels_in_config:
                x, y, xmax, ymax = float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': x / img_width * 100,
                        'y': y / img_height * 100,
                        'width': (xmax - x) / img_width * 100,
                        'height': (ymax - y) / img_height * 100
                    },
                    'score': float(row['confidence'])
                })
                all_scores.append(float(row['confidence']))

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        return [{
            'result': results,
            'score': avg_score
        }]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data