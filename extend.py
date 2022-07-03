from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
class TemporalReduceLROnPlateau(ReduceLROnPlateau):
    #后期收敛变慢,学习率减半速率也应该变慢也就是patience增大
    def __init__(self,
               monitor='val_loss',
               factor=0.1,
               patience=5,
               patience_growth_per_epoch=0.1,
               max_patience=30,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
                **kwargs):
        self.max_patience=max_patience
        self.patience_growth_per_epoch=patience_growth_per_epoch
        super(TemporalReduceLROnPlateau,self).__init__(monitor,factor,patience,verbose,mode,min_delta,cooldown,min_lr)
    def on_epoch_end(self, epoch, logs=None):
        super(TemporalReduceLROnPlateau,self).on_epoch_end(epoch,logs)
        if self.patience<self.max_patience:
            self.patience=self.patience+self.patience_growth_per_epoch
            if int(self.patience)!=int(self.patience+self.patience_growth_per_epoch):
                print('\nEpoch %05d Increase patience to %.f'%(epoch+1,self.patience))
class WarmupExponentialDecay(Callback):
    def __init__(self,lr_base=0.0002,lr_min=0.0,decay=0,warmup_epochs=0):
        self.num_passed_batchs = 0   #一个计数器
        self.warmup_epochs=warmup_epochs  
        self.lr=lr_base #learning_rate_base
        self.lr_min=lr_min #最小的起始学习率,此代码尚未实现
        self.decay=decay  #指数衰减率
        self.steps_per_epoch=0 #也是一个计数器
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch==0:
            #防止跑验证集的时候呗更改了
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        K.get_value(self.model.optimizer.lr)*(1-self.decay))
        self.num_passed_batchs += 1
    def on_epoch_begin(self,epoch,logs=None):
    #用来输出学习率的,可以删除
        print("learning_rate:",K.get_value(self.model.optimizer.lr))         
class DecoupleWeightDecay:
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        weight_decay: weight decay value that will be mutltiplied to the parameter
    # References
        - [AdamW - DECOUPLED WEIGHT DECAY REGULARIZATION](
           https://arxiv.org/pdf/1711.05101.pdf)
    """

    def __init__(self, weight_decay, **kwargs):
        with K.name_scope(self.__class__.__name__):
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
        super(DecoupleWeightDecay, self).__init__(**kwargs)

    def get_updates(self, loss, params):
        updates = super(DecoupleWeightDecay, self).get_updates(loss, params)
        #TODO change loop to vectorized
        for p in params:
            updates.append(K.update_sub(p, self.weight_decay*p))
        return updates


def create_decouple_optimizer(optimizer):
    class OptimizerW(DecoupleWeightDecay, optimizer):
        def __init__(self, weight_decay, **kwargs):
            super(OptimizerW, self).__init__(weight_decay, **kwargs)


class WeightDecayScheduler(Callback):
    def __init__(self, init_lr):
        super(WeightDecayScheduler, self).__init__()
        self.previous_lr = init_lr


    def on_epoch_begin(self, epoch, logs=None):
        current_lr = float(K.get_value(self.model.optimizer.lr))
        coeff = current_lr / self.previous_lr
        new_weight_decay = float(K.get_value(self.model.optimizer.weight_decay)) * coeff
        K.set_value(self.model.optimizer.weight_decay, new_weight_decay)
        self.previous_lr = current_lr
        if coeff!=1:
            print(epoch, coeff)

    def on_epoch_end(self, epoch, logs=None):
        return
class AdamW(DecoupleWeightDecay, Adam):
    def __init__(self, weight_decay, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(AdamW, self).__init__(weight_decay=weight_decay, lr=lr, beta_1=beta_1, beta_2=beta_2,
                 epsilon=epsilon, decay=decay, amsgrad=amsgrad, **kwargs)