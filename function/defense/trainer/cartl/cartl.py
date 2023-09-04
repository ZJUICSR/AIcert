import os
from .src import settings
from .src.utils import logger
from .src.attack import LinfPGDAttack
from .src.trainer import RobustPlusSingularRegularizationTrainer, SpectralNormTransferLearningTrainer
from .src.cli.utils import get_model, get_test_dataset, get_train_dataset
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def fdm(model, num_classes, dataset, random_init, epsilon, step_size, num_steps, k, lambda_):
    """Cooperative Adversarially-Robust TransferLearning"""
    save_name = f"cartl_{model}_{dataset}_{k}_{lambda_}"
    if not os.path.exists(settings.log_dir):
            os.makedirs(settings.log_dir)
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    params = {
        "random_init": random_init,
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "dataset_name": dataset
    }
    trainer = RobustPlusSingularRegularizationTrainer(
        k=k,
        _lambda=lambda_,
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        attacker=LinfPGDAttack,
        params=params,
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    return trainer.train(f"{settings.model_dir / save_name}")

def sn_tl(model, num_classes, dataset, k, teacher, power_iter, norm_beta, freeze_bn, reuse_statistic, reuse_teacher_statistic):
    """transform leanring"""
    from .utils import make_term
    term = make_term(freeze_bn, reuse_statistic, reuse_teacher_statistic)
    save_name = f"sntl_{power_iter}_{norm_beta}_{term}_{model}_{dataset}_{k}_{teacher}_{settings.seed}"

    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    trainer = SpectralNormTransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),\
        model=get_model(model=model, num_classes=num_classes, k=k),
        train_loader=get_train_dataset(dataset=dataset),
        test_loader=get_test_dataset(dataset=dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        power_iter=power_iter,
        norm_beta=norm_beta,
        freeze_bn=freeze_bn,
        reuse_statistic=reuse_statistic,
        reuse_teacher_statistic=reuse_teacher_statistic,
    )

    return trainer.train(f"{settings.model_dir / save_name}")


def cartl(source_dataset, source_num_classes, target_dataset, target_num_classes):
    model = 'res18'
    epsilon = 8/255
    step_size = 2/255
    num_steps = 7
    source_k = 6
    lambda_ = 0.01
    random_init = True

    fdm(
        model=model, 
        num_classes=source_num_classes, 
        dataset=source_dataset, 
        random_init=random_init, 
        epsilon=epsilon, 
        step_size=step_size, 
        num_steps=num_steps,
        k=source_k,
        lambda_=lambda_, 
    )

    target_k = 6
    teacher = 'cartl_res18_cifar100_6_0.01-best_robust'
    power_iter = 1
    norm_beta = 1.0
    freeze_bn = False
    reuse_statistic = False
    reuse_teacher_statistic = False

    return sn_tl(model=model,
        num_classes=target_num_classes,
        dataset=target_dataset,
        k=target_k,
        teacher=teacher,
        power_iter=power_iter,
        norm_beta=norm_beta,
        freeze_bn=freeze_bn,
        reuse_statistic=reuse_statistic,
        reuse_teacher_statistic=reuse_teacher_statistic
    )