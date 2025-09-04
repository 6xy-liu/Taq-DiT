import logging
from .fake_quant import QuantizeBase,GELULSQFakeQuantize
logger = logging.getLogger("quantization")


def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.info('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.info('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_quantization(model, quantizer_type='fake_quant'):
    logger.info('Disable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.info('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.info('Disable observer and Enable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant()


def disable_all(model):
    logger.info('Disable observer and disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            logger.info('Disable observer and disable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()
            if isinstance(submodule, GELULSQFakeQuantize):
                logger.info('disable shift update: {}'.format(name))
                submodule.disable_shift_value_updata()             

def inference_all(model):
    logger.info('enable all with dynamic quant.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if 'fake_quant' in name:
                logger.info('normal_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.enable_fake_quant()
            else:
                logger.info('dynamic_qaunt: {}'.format(name))
                submodule.enable_observer()
                submodule.enable_fake_quant()

def debug_all(model):
    logger.info('enable all with dynamic quant.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if 'block_post_act_fake_quantize_glue_out' in name:
                logger.info('dynamic_qaunt: {}'.format(name))
                submodule.enable_observer()
                submodule.enable_fake_quant()                
            else:
                logger.info('normal_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()

def enable_shift_value_update(model):
    logger.info('enable post-GELU shift value update.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, GELULSQFakeQuantize):
            logger.info('enable shift update: {}'.format(name))
            submodule.enable_shift_value_updata()

def disable_shift_value_update(model):
    logger.info('disable post-GELU shift value update.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, GELULSQFakeQuantize):
            logger.info('disable shift update: {}'.format(name))
            submodule.disable_shift_value_updata() 
