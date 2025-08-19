# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
# 基于submitit库的多节点分布式训练作业提交脚本
# 专门用于在集群环境（如使用 Slurm 调度系统的集群）中启动 MAE（掩码自编码器）的大规模预训练任务。
# 简化集群资源配置、分布式训练初始化和作业管理，确保多节点、多 GPU 的训练任务能在集群上稳定运行。
import argparse
import os
import uuid
from pathlib import Path

import main_pretrain as trainer
import submitit

# 在原有 MAE 预训练参数（如批次大小、模型类型等）的基础上，增加了集群作业相关的参数：
def parse_args():
    # 1. 获取MAE预训练的基础参数解析器（包含模型、训练、数据等核心参数）
    # 这里的trainer指之前定义的MAE预训练模块（如main_pretrain.py）
    trainer_parser = trainer.get_args_parser()
    # 2. 创建新的参数解析器，用于提交MAE预训练的集群作业
    # parents=[trainer_parser]表示继承基础参数，避免重复定义
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    # 3. 添加集群节点与GPU资源相关的参数
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node") # 每个节点请求的GPU数量
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request") # 请求的节点总数
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job") # 作业超时时间（单位：分钟，默认4320分钟=3天）
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.") # 作业输出目录（留空则自动生成）

    # 4. 添加集群调度相关的参数
    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit") # 提交作业的集群分区（队列名称）
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs") # 是否请求32GB显存的Volta架构GPU（如V100-32G）
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler") # 提交给调度器的作业备注（如实验说明）
    # 5. 解析命令行输入的参数并返回
    return parser.parse_args()


# 获取集群中的共享文件夹路径（如/checkpoint/用户名/experiments），用于多节点间共享训练日志、检查点等文件
def get_shared_folder() -> Path:
    # 获取当前系统登录的用户名（从环境变量中读取）
    user = os.getenv("USER")
    # 检查集群中是否存在/checkpoint/目录（常见于分布式训练集群的共享存储路径）
    if Path("/checkpoint/").is_dir():
        # 定义共享文件夹路径：/checkpoint/用户名/experiments
        # 用于存储多节点训练时的共享文件（如初始化文件、检查点等）
        p = Path(f"/checkpoint/{user}/experiments")
        # 创建该文件夹（如果不存在），exist_ok=True确保文件夹已存在时不报错
        p.mkdir(exist_ok=True)
        # 返回创建的共享文件夹路径
        return p
    # 如果没有找到可用的共享文件夹，抛出运行时错误
    raise RuntimeError("No shared folder available")

# 生成分布式训练的初始化文件（UUID 唯一标识），用于多节点进程组的初始化（解决分布式通信的 “握手” 问题）
def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    # 注释说明：分布式训练的初始化文件必须不存在（确保每次启动都是新的初始化），但其父目录必须存在
    # 确保共享文件夹存在（调用get_shared_folder()获取路径，并创建父目录，已存在则不报错）
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    # 生成唯一的初始化文件路径：在共享文件夹下，用UUID生成唯一文件名（十六进制字符串）+ "_init"后缀
    # UUID确保每次调用生成不同的文件名，避免多任务冲突
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    # 检查该初始化文件是否已存在（极端情况，如UUID重复），若存在则删除
    if init_file.exists():
        os.remove(str(init_file))
    # 返回生成的初始化文件路径（用于分布式训练的进程组初始化）
    return init_file

# 连接集群作业与 MAE 训练逻辑的核心
class Trainer(object):
    # 初始化方法：保存命令行参数到实例变量
    def __init__(self, args):
        self.args = args  # 将解析后的参数对象存储为实例属性，供其他方法使用

    # 实例调用方法：当Trainer实例被提交给submitit时，自动执行此方法（核心训练逻辑入口）
    def __call__(self):
        # 导入MAE预训练的主模块（避免循环导入，在需要时才导入）
        import main_pretrain as trainer

        # 配置GPU相关的分布式参数（如进程排名、设备编号等）
        self._setup_gpu_args()
        # 调用MAE预训练的主函数，传入配置好的参数，启动训练
        trainer.main(self.args)

    # 检查点方法：用于作业超时或中断时的重启逻辑（submitit会自动调用）
    def checkpoint(self):
        # 导入必要模块（在需要时导入，减少初始化开销）
        import os
        import submitit

        # 重新生成分布式初始化文件的URL（确保新作业使用新的初始化文件）
        self.args.dist_url = get_init_file().as_uri()
        # 定义检查点文件路径（训练过程中保存的模型断点）
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        # 若检查点文件存在，设置resume参数为该文件路径（从断点继续训练）
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        # 打印重启信息，便于调试
        print("Requeuing ", self.args)
        # 创建新的Trainer实例（复用当前参数）
        empty_trainer = type(self)(self.args)
        # 返回延迟提交对象，告知submitit重启作业
        return submitit.helpers.DelayedSubmission(empty_trainer)

    # 配置GPU和分布式训练的参数（适配集群作业环境）
    def _setup_gpu_args(self):
        # 导入必要模块
        import submitit
        from pathlib import Path

        # 获取集群作业的环境信息（由submitit提供，包含进程排名、节点信息等）
        job_env = submitit.JobEnvironment()
        # 更新输出目录：将路径中的"%j"替换为实际作业ID（确保每个作业有唯一输出目录）
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        # 日志目录与输出目录保持一致
        self.args.log_dir = self.args.output_dir
        # 设置当前进程使用的本地GPU编号（在节点内的相对编号，如0-7）
        self.args.gpu = job_env.local_rank
        # 设置当前进程的全局排名（在所有节点的所有进程中的唯一编号）
        self.args.rank = job_env.global_rank
        # 设置总进程数（所有节点的GPU数量之和，即world_size）
        self.args.world_size = job_env.num_tasks
        # 打印进程组信息，便于监控分布式训练状态
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

# 提交集群作业
def main():
    # 解析命令行参数（包括MAE预训练参数和集群作业参数）
    args = parse_args()
    # 若未指定作业目录，自动生成路径（共享文件夹下以作业ID为标识）
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"  # %j会被实际作业ID替换

    # Note that the folder will depend on the job_id, to easily track experiments
    # 注意：文件夹路径会依赖作业ID，便于区分和追踪不同实验
    # 初始化submitit执行器，指定作业输出文件夹和最大超时重试次数（30次）
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # 从参数中提取集群资源配置
    num_gpus_per_node = args.ngpus  # 每个节点的GPU数量
    nodes = args.nodes              # 节点总数
    timeout_min = args.timeout      # 作业超时时间（分钟）

    # 集群分区（队列）配置
    partition = args.partition
    kwargs = {}  # 用于存储额外的集群调度参数
    # 若请求32G显存的Volta GPU，添加Slurm约束
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    # 若有作业备注，添加到Slurm参数中
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    # 更新执行器的资源参数（向集群申请资源）
    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,  # 每个节点的内存（40GB * GPU数量）
        gpus_per_node=num_gpus_per_node,  # 每个节点的GPU数量
        tasks_per_node=num_gpus_per_node,  # 每个节点的任务数（1个任务对应1个GPU）
        cpus_per_task=10,  # 每个任务分配的CPU核心数
        nodes=nodes,  # 总节点数
        timeout_min=timeout_min,  # 作业超时时间（最大为60*72=4320分钟）
        # 以下是集群相关的特定参数
        slurm_partition=partition,  # 指定提交的集群分区
        slurm_signal_delay_s=120,  # 作业超时前120秒发送信号，用于保存检查点
        **kwargs  # 附加其他调度参数（如GPU约束、备注）
    )

    # 设置作业名称为"mae"（便于在集群中识别）
    executor.update_parameters(name="mae")

    # 配置分布式训练的初始化文件URL（多节点通信的关键）
    args.dist_url = get_init_file().as_uri()
    # 设置输出目录为作业目录（与submitit的输出文件夹一致）
    args.output_dir = args.job_dir

    # 创建Trainer实例（封装了MAE训练逻辑和集群适配）
    trainer = Trainer(args)
    # 提交作业到集群，返回作业对象
    job = executor.submit(trainer)

    # 打印提交的作业ID（用于追踪作业状态）
    # print("Submitted job_id:", job.job_id)
    print(job.job_id)


if __name__ == "__main__":
    main()
