from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
import os
import logging
import _pickle as cPickle


class Config:
	"""
	All the parameters/settings of the entire project
	"""

	RANDOM_SEED = 42
	EXPERIMENT_NAME = 'ko_ocr'
	ROOT_OUT_FOLDER = '../outputs'
	OUPUT_FOLDER = os.path.join(ROOT_OUT_FOLDER, EXPERIMENT_NAME)
	OUTFILE = 'bi-stet_ko_ocr.cp'
	LOG_FNAME = 'logging.log'
	SAMPLES_PER_BATCH = 64

	def __init__(self):
		self.DEBUG = False
		# INPUT/OUTPUT PARAMETERS

		self.MAX_SEQ_LENGTH = 24
		self.IMAGE_SHAPE = (32, 224)

		self.INPUT_TRANSFORM = Compose([
			Resize(self.IMAGE_SHAPE),
			ToTensor(),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		# TRAIN DATASETS

		self.LOAD_PICKLE = False

		self.KO_OCR_TRAIN_ROOT = '/flute/data/ko_ocr/train/cropped/'
		self.KO_OCR_TRAIN_ANNOTATIONS = 'annotations.txt'

		self.LOG_FNAME = self.LOG_FNAME

		self.TRAIN_DATASETS = [
			(self.KO_OCR_TRAIN_ROOT, self.KO_OCR_TRAIN_ANNOTATIONS)
		]
	
		# TEST DATASETS

		self.KO_OCR_VALID = '/flute/data/ko_ocr/valid/cropped'
		self.KO_OCR_VALID_ANNOTATIONS = 'annotations.txt'

		# OUTPUT FOLDERS

		self.EXPERIMENT_NAME = self.EXPERIMENT_NAME
		self.ROOT_OUT_FOLDER = self.ROOT_OUT_FOLDER
		self.OUPUT_FOLDER = self.OUPUT_FOLDER
		self.OUTFILE = self.OUTFILE

		# TRAINING PARAMS

		self.NUM_WORKERS = 0
		self.RANDOM_SEED = self.RANDOM_SEED
		self.SAMPLES_PER_BATCH = self.SAMPLES_PER_BATCH
		self.TRAIN_ITERATIONS = 500000
		self.SUMMARY_INTERVAL = 200
		self.STORE_ITERVAL = 5000
		self.START_ITERATION = 0

		# MODEL PARAMS
		self.BIDIRECTIONAL_DECODING = True
		self.N = 6
		self.D_MODEL = 512
		self.D_FF = 2048
		self.H = 8
		self.DROPOUT = 0.1
		self.USE_RESNET = True
		self.RESNET_LAYERS = [3, 4, 6, 6, 3]

		# OPTIMIZER

		self.LEARNING_RATE = 1

		self.SIZE_AVARAGE = True
		self.REDUCE_LOSS = False
		self.WARMUP = 8000
		self.FACTOR = 1

		# LEARNING RATE SCHEDUELER

		self.LR_MILESTONES = [150000, 300000, 400000]
		self.LR_GAMMA = 0.1

		self.LOAD_MODEL = False
		self.VALIDATE = True

		self.EVAL_ONLY = True

		self.CONFIG_FNAME = 'config.cpkl'
		self.MODEL_FILE = None

		if self.MODEL_FILE is not None and not self.VALIDATE:
			assert self.START_ITERATION > 1
	
	def store_config(self, path):
		"""
		
		:param path:
		:return:
		"""
		
		f = open(os.path.join(path, self.CONFIG_FNAME), 'wb')
		cPickle.dump(self.__dict__, f)
		f.close()
	
	def load_config(self, path):
		"""
		
		:param path: path to load the config from
		:return:
		"""
		
		with open(os.path.join(path, self.CONFIG_FNAME), 'rb') as f:
			self.__dict__.update(cPickle.load(f))

	def print_config(self, print_on_std=True, store=False):
		"""
		:param print_on_std: If true print
		:param store: store the config in a .txt file
		:return:
		"""
		
		if store:
			f = open(os.path.join(Config.OUPUT_FOLDER, 'config.txt'), 'w')
			
		for key, value in self.__dict__.items():
			if not(key[:2] == '__'):
				line = str(key) + " : " + str(value)
				if print_on_std:
					logging.info(line)
				if store:
					f.write(line + "\n")
		if store:
			f.close()

	def set_validation_config(self):

		self.VALIDATE = True
		self.MODEL_FILE = 'bi-stet_ko_ocr.cp'
		self.LOAD_MODEL = True
		self.EXPERIMENT_NAME = 'valid'
		self.ROOT_OUT_FOLDER = '/flute/outputs/ko_ocr'

		self.OUPUT_FOLDER = os.path.join(self.ROOT_OUT_FOLDER, self.EXPERIMENT_NAME)
		self.LEXICON_INFERENCE = True
		self.SAMPLES_PER_BATCH = 1
		self.LOAD_PICKLE = False
		self.LOG_FNAME = 'validation.log'
		self.FILTER_DIFFICULT = False
		self.SHOW_EXAMPLES = False
		self.WRITE_EXAMPLES = True
		self.WORD_LENGTH_ACCURACY = True
		self.BIDIRECTIONAL_DECODING = True

		self.VALIDATION_DATASETS = [
			(self.KO_OCR_VALID, self.KO_OCR_VALID_ANNOTATIONS, 'ko_ocr_valid')
		]