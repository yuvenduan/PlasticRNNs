"""
Configurations for the project
format adapted from https://github.com/gyyang/olfaction_evolution

Be aware that the each field in the configuration must be in basic data type that
jason save and load can preserve. each field cannot be complicated data type
"""

class BaseConfig(object):

    def __init__(self):
        """
        model_type: model type, eg. "ConvRNNBL"
        task_type: task type, eg. "n_back"
        """
        self.experiment_name = None
        self.model_name = None
        self.task_type = None
        self.dataset = None
        self.save_path = None

        self.seed = 0

        # Weight for each dataset
        self.mod_w = [1.0, ]

        # Specify required resources (useful when running on cluster)
        self.hours = 24
        self.mem = 32
        self.cpu = 1 
        self.num_workers = 1

        # Refer to https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-submit-GPU-jobs%3F for detailed explanation
        self.gpu_constraint = 'high-capacity' 

        # basic evaluation parameters
        self.store_states = False

        # if not None, load model from the designated path
        self.load_path = None

        # if overwrite=False and training log is complete, skip training
        self.overwrite = True

        self.config_mode = 'train'
        self.training_mode = 'supervised'

        # plasticity
        self.plasticity_mode = 'gradient'
        self.inner_lr_mode = 'random'
        self.p_lr = 0.1
        self.p_wd = 0.1
        self.inner_grad_clip = 1
        self.extra_dim = 4
        self.extra_input_dim = 0
        self.modulation = True
        self.random_network = False

        self.input_shape = (1, )
        # output size of the model, which is the number of classes in classification tasks
        self.model_outsize = 10 
        
        self.hidden_size = 128
        self.layernorm = False
        self.weight_clip = None

    @property
    def test_begin(self):
        return self.seq_length + self.delay

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

class SupervisedLearningBaseConfig(BaseConfig):

    def __init__(self):
        super().__init__()

        self.training_mode = 'supervised'

        # max norm of grad clipping, eg. 1.0 or None
        self.grad_clip = 5

        # optimizer
        self.batch_size = 20
        self.optimizer_type = 'AdamW'
        self.lr = 0.001
        self.wdecay = 0.0003

        # scheduler
        self.use_lr_scheduler = False
        self.scheduler_type = None

        # training
        self.num_ep = 100000
        self.max_batch = 5000

        # evaluation
        self.perform_val = True
        self.perform_test = True
        self.log_every = 200
        self.save_every = 100000
        self.test_batch = 100

        self.print_mode = 'accuracy' # or print error
        self.model_type = 'SimpleRNN'
        self.rnn = 'RNN'

        self.cnn = None
        self.cnn_pretrain = 'none'
        self.freeze_pretrained_cnn = True
        self.pretrain_step = 10000

        self.do_analysis = False

class CueRewardConfig(SupervisedLearningBaseConfig):

    def __init__(self):
        super().__init__()

        self.task_type = 'SeqRegression'
        self.dataset = 'CueReward'

        self.input_noise = 0.1
        self.model_outsize = 1
        self.input_shape = (16, )
        self.seq_length = self.delay = 0

        self.trial_length = 20
        self.n_class = 5
        self.max_batch = 20000

        self.model_type = 'SimpleRNN'
        self.hidden_size = 128
        
        self.wdecay = 0
        self.batch_size = 64
        self.print_mode = 'error'

class SeqReproductionConfig(SupervisedLearningBaseConfig):

    def __init__(self):
        super().__init__()

        self.task_type = 'SeqRegression'
        self.dataset = 'SeqReproduction'

        self.input_shape = (3, )
        self.seq_length = 20
        self.delay = 20
        self.model_outsize = 1

        self.model_type = 'SimpleRNN'
        self.hidden_size = 128
        
        self.wdecay = 0
        self.batch_size = 64
        self.max_batch = 10000
        self.print_mode = 'error'

class RegressionConfig(SupervisedLearningBaseConfig):

    def __init__(self):
        super().__init__()

        self.task_type = 'SeqRegression'
        self.dataset = 'Regression'

        self.seq_length = 0
        self.delay = 0
        self.train_length = 20
        self.test_length = 20

        self.input_shape = (12, )
        self.input_noise = 0.1
        self.model_outsize = 1

        self.model_type = 'SimpleRNN'
        self.hidden_size = 128
        self.task_mode = 'linear'
        
        self.wdecay = 0
        self.batch_size = 64
        self.max_batch = 10000
        self.print_mode = 'error'

class FSCConfig(SupervisedLearningBaseConfig):

    def __init__(self):
        super().__init__()

        self.task_type = 'FSC'
        self.dataset = 'FSC'

        self.image_dataset = 'miniImageNet'
        self.input_shape = (3, 84, 84)

        self.train_way = 5
        self.train_shot = 1
        self.train_query = 1
        self.model_outsize = 5
        self.extra_input_dim = 5

        self.randomize_train_order = True
        self.label_smoothing = 0
        
        self.test_batch = 200
        self.log_every = 1000
        self.max_batch = 40000
        self.hours = 24
        self.gpu_constraint = '11GB'
        self.use_lr_scheduler = True
        self.scheduler_type = 'CosineAnnealing'

        self.model_type = 'CNNtoRNN'
        self.cnn = 'ProtoNet'
        self.hidden_size = 256

        self.cnn_pretrain = 'none'
        self.pretrain_step = 'best'
        self.freeze_pretrained_cnn = True

        self.perform_val = True
        self.perform_test = True

        self.wdecay = 5e-4
        self.lr = 1e-3
        self.batch_size = 64

        self.print_mode = 'accuracy'

class ClassificationPretrainConfig(SupervisedLearningBaseConfig):

    def __init__(self):
        super().__init__()

        self.task_type = 'Classification'
        self.dataset = 'Image'

        self.batch_size = 128
        self.eval_iter = 100

        self.max_batch = 20000
        self.log_every = 500

        self.lr = 3e-3
        self.wdecay = 1e-4

        self.image_dataset = 'miniImageNet'
        self.input_shape = (3, 84, 84)

        self.model_type = 'CNN'
        self.cnn = 'ProtoNet'
        self.perform_val = True

        self.use_lr_scheduler = True
        self.scheduler_type = 'CosineAnnealing'

class ContrastivePretrainConfig(SupervisedLearningBaseConfig):

    def __init__(self):
        super().__init__()

        self.model_type = 'CNN'
        self.use_mlp_proj = True
        self.model_out_size = 256

        self.task_type = 'Contrastive'
        self.dataset = 'SimCLR'

        self.image_dataset = 'miniImageNet'
        self.batch_size = 128

        self.max_batch = 100000
        self.log_every = 200

        self.perform_val = self.perform_test = False
        self.wdecay = 1e-4

        self.use_lr_scheduler = False
        self.lr = 1e-3

        self.cpu = 4
        self.num_workers = 4
        self.hours = 24