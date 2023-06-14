# TODO
# 避免 options 重复

import functools
from typing import Callable, Iterable, Reversible, Union

import click

import torch

from .utils import (DefaultDataset, SupportDatasetList,
                    DefaultModel, SupportModelList,
                    SupportParsevalModelList, SupportNormalModelList,
                    get_test_dataset, get_train_dataset,
                    get_model)

from ...src import settings

from ...src.utils import logger

from ...src.trainer import (TransferLearningTrainer, LWFTransferLearningTrainer,
                         RetrainTrainer, NormalTrainer, ADVTrainer, 
                         RobustPlusSingularRegularizationTrainer,
                         BNTransferLearningTrainer, SpectralNormTransferLearningTrainer)

from ...src.attack import LinfPGDAttack

_BasicOptions = [
    click.option("-m", "--model", type=click.Choice(SupportModelList),
                 default=DefaultModel, show_default=True, help="neural network"),
    click.option("-n", "--num_classes", type=int,
                 default=10, show_default=True, help="number of classes"),
    click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
                 default=DefaultDataset, show_default=True, help="dataset"),
]


@click.group()
def cli():
    ...


def apply_options(options: Union[Iterable, Reversible]):
    def _decorators(f: Callable):
        @functools.wraps(f)
        def _apply():
            nonlocal f
            for option in reversed(options):
                f = option(f)
            return f

        return _apply

    return _decorators


def composed(decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def apply_test(f):
    return composed(_BasicOptions)(f)


@cli.command()
# @apply_options(_BasicOptions)
@click.option("-m", "--model", type=click.Choice(SupportModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-t", "--teacher", type=str, required=True,
              help="filename of teacher model")
def tl(model, num_classes, dataset, k, teacher):
    """transform leanring"""
    save_name = f"tl_{model}_{dataset}_{k}_{teacher}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    trainer = TransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),
        model=get_model(model, num_classes, k),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportNormalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-l", "--lambda_", type=float, required=True,
              help="penalization rate of feature representation between teacher and student")
@click.option("-t", "--teacher", type=str, required=True,
              help="filename of teacher model")
def lwf(model, num_classes, dataset, lambda_, teacher):
    """learning without forgetting"""
    save_name = f"lwf_{model}_{dataset}_{lambda_}_{teacher}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    trainer = LWFTransferLearningTrainer(
        _lambda=lambda_,
        teacher_model_path=str(settings.model_dir / teacher),
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-t", "--teacher", type=str, required=True,
              help="filename of teacher model")
@click.option("-fb", "--freeze-bn", is_flag=True, help="freeze bn layer", show_default=True)
@click.option("-rs", "--reuse-statistic", is_flag=True, help="reuse statistic", show_default=True)
@click.option("-rts", "--reuse-teacher-statistic", is_flag=True, help="reuse teacher statistic", show_default=True)
def bntl(model, num_classes, dataset, k, teacher, freeze_bn, reuse_statistic, reuse_teacher_statistic):
    """normal transfer learning with batch norm operations"""
    save_name = f"bntl_{model}_{dataset}_{k}_{teacher}" \
                f"{'_fb' if freeze_bn else ''}" \
                f"{'_rs' if reuse_statistic else ''}" \
                f"{'_rts' if reuse_teacher_statistic else ''}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    trainer = BNTransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),
        model=get_model(model, num_classes, k),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        freeze_bn=freeze_bn,
        reuse_statistic=reuse_statistic,
        reuse_teacher_statistic=reuse_teacher_statistic
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportParsevalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-t", "--teacher", type=str, required=True,
              help="filename of teacher model")
@click.option("-pt", "--power-iter", type=int,
              default=1, show_default=True, help="number of power iterations to calculate spectral norm")
@click.option("-nt", "--norm-beta", type=float,
              default=1.0, show_default=True, help="norm beta, definition can be found in paper (11)")
@click.option("-fb", "--freeze-bn", is_flag=True, help="freeze bn layer", show_default=True)
@click.option("-rs", "--reuse-statistic", is_flag=True, help="reuse statistic", show_default=True)
@click.option("-rts", "--reuse-teacher-statistic", is_flag=True, help="reuse teacher statistic", show_default=True)
def sntl(model, num_classes, dataset, k, teacher,
         power_iter, norm_beta, freeze_bn, reuse_statistic, reuse_teacher_statistic):
    """transfer learning with spectrum norm"""
    save_name = f"sntl_{power_iter}_{norm_beta}_" \
                f"{'True' if freeze_bn else 'False'}_" \
                f"{'rts_' if reuse_teacher_statistic else ''}" \
                f"{'rs_' if reuse_statistic else ''}" \
                f"{model}_{dataset}_{k}_{teacher}_{settings.seed}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    trainer = SpectralNormTransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),
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

    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportNormalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("-k", "--k", type=int, required=True,
              help="trainable blocks from last")
@click.option("-st", "--state_dict", type=str, required=True,
              help="filename of state dict for model to be retrained")
def nr(model, num_classes, dataset, k, state_dict):
    """normal retrain"""
    save_name = f"nr_{model}_{dataset}_{k}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    model = get_model(model, num_classes, k)
    model.load_state_dict(torch.load(str(settings.model_dir / state_dict)))
    trainer = RetrainTrainer(
        k=k,
        model=model,
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportNormalModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
def nt(model, num_classes, dataset):
    """normal train"""
    save_name = f"nt_{model}_{dataset}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    trainer = NormalTrainer(
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("--random_init/--no_random_init", default=True,
              show_default=True, help="PGD/BIM")
@click.option("-e", "--epsilon", type=float, default=8 / 255,
              show_default=True, help="epsilon")
@click.option("-ss", "--step_size", type=float, default=2 / 255,
              show_default=True, help="step size")
@click.option("-ns", "--num_steps", type=int, default=7,
              show_default=True, help="num steps")
def at(model, num_classes, dataset, random_init, epsilon, step_size, num_steps):
    """adversarial train"""
    save_name = f"at_{model}_{dataset}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    params = {
        "random_init": random_init,
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "dataset_name": dataset
    }
    trainer = ADVTrainer(
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        attacker=LinfPGDAttack,
        params=params,
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


@cli.command()
@click.option("-m", "--model", type=click.Choice(SupportModelList),
              default=DefaultModel, show_default=True, help="neural network")
@click.option("-n", "--num_classes", type=int,
              default=10, show_default=True, help="number of classes")
@click.option("-d", "--dataset", type=click.Choice(SupportDatasetList),
              default=DefaultDataset, show_default=True, help="dataset")
@click.option("--random_init/--no_random_init", default=True,
              show_default=True, help="PGD/BIM")
@click.option("-e", "--epsilon", type=float, default=8 / 255,
              show_default=True, help="epsilon")
@click.option("-ss", "--step_size", type=float, default=2 / 255,
              show_default=True, help="step size")
@click.option("-ns", "--num_steps", type=int, default=7,
              show_default=True, help="num steps")
@click.option("-k", "--k", type=int, required=True,
              help="kth(from last) layer norm will be used in loss")
@click.option("-l", "--lambda_", type=float, required=True,
              help="penalization rate of layer norm")
def fdm(model, num_classes, dataset, random_init, epsilon, step_size, num_steps, k, lambda_):
    """Feature Distance Minimization"""
    save_name = f"cartl_{model}_{dataset}_{k}_{lambda_}"
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
    trainer.train(f"{settings.model_dir / save_name}")


cli = cli()
