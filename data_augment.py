import cv2
import numpy as np
import copy


def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img_one = cv2.imread(img_data_aug['filepath'].split()[0])
	img_two = cv2.imread(img_data_aug['filepath'].split()[1])

	if augment:
		rows, cols = img_one.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img_one = cv2.flip(img_one, 1)
			img_two = cv2.flip(img_two, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img_one = cv2.flip(img_one, 0)
			img_two = cv2.flip(img_two, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img_one = np.transpose(img_one, (1,0,2))
				img_one = cv2.flip(img_one, 0)

				img_two = np.transpose(img_two, (1, 0, 2))
				img_two = cv2.flip(img_two, 0)
			elif angle == 180:
				img_one = cv2.flip(img_one, -1)
				img_two = cv2.flip(img_two, -1)
			elif angle == 90:
				img_one = np.transpose(img_one, (1,0,2))
				img_one = cv2.flip(img_one, 1)

				img_two = np.transpose(img_two, (1, 0, 2))
				img_two = cv2.flip(img_two, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug['width'] = img_one.shape[1]
	img_data_aug['height'] = img_two.shape[0]
	return img_data_aug, img_one, img_two
