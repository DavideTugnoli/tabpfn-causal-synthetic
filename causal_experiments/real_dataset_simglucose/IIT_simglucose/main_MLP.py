import argparse
from datetime import datetime
import json
import os
from pickle import dump
import torch
from dataset.glucosedataset import setup_loaders
from models.teacher import simglucose as teacher
from models.teacher import simglucose_no_cycles as teacher_no_cycles
from models.student import MLP as student
from models.student import MLP_tree as student_tree
from models.student import MLP_tree_depth as student_tree_depth
from models.student import MLP_tree_joint as student_tree_joint
from models.student import MLP_scaled as student_scaled
from utils.counterfactual_utils import set_seed, logger
from utils.trainer import Trainer

def prepare_trainer(args):

    # ARGS #
    set_seed(args)

    torch.set_num_threads(os.cpu_count())

    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

    # SAVE PARAMS #
    logger.info(f"Param: {args}")
    with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.student_model == "parallel":
        student_model = student.MLP()
    elif args.student_model == "tree":
        student_model = student_tree.MLP()
    elif args.student_model == "tree_depth":
        student_model = student_tree_depth.MLP()
    elif  args.student_model == "tree_joint":
        student_model = student_tree_joint.MLP()
    elif args.student_model == "scaled":
        # EX: Pred horizon 30 with a time step of 3 minutes represent 10 integrations in the simulator --> 1 initial block + 9 scaled
        student_model = student_scaled.MLP_scaled(args.input_size, args.output_size, args.pred_horizon/3)
    # student = student_model.to(f"cuda:0", non_blocking=True)
    logger.info("Student loaded.")

    if args.modified:
        teacher_model = teacher_no_cycles.Simglucose(args.pred_horizon)
    else:
        teacher_model = teacher.Simglucose(args.pred_horizon)
    # teacher = teacher_model.to(f"cuda:0", non_blocking=True)
    logger.info("Teacher loaded.")

    # DATA LOADER
    train_dataset, val_dataset, test_dataset = setup_loaders(args)
    logger.info("Data loader created.")

    # TRAINER #
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    trainer = Trainer(
        params=args,
        dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        neuro_mapping=args.neuro_mapping,
        student=student_model,
        teacher=teacher_model
    )
    logger.info("trainer initialization done.")
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--modified",
        type=bool,
        default=False,
        help="Modified true for the teacher model without cycles"
    )
    parser.add_argument(
        "--student_model",
        type=str,
        choices=["parallel", "tree", "tree_depth", "tree_joint", "scaled"],
        help="Prediction horizon."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=56,
        help="Random seed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite dump_path if it already exists."
    )
    parser.add_argument(
        "--neuro_mapping",
        type=str,
        # default="train_config/MLP_parallel.nm",
        # default=None,
        help="Predefined neuron mapping for the interchange experiment.",
    )
    parser.add_argument(
        "--alpha_ce",
        type=float,
        default=0.25,
        help="Coefficient regular loss",
    )
    parser.add_argument(
        "--alpha_causal",
        type=float,
        default=0.75,
        help="Coefficient causal loss",
    )
    parser.add_argument("--n_epoch", type=int, default=300, help="Number of epochs.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=20,
        help="Gradient accumulation for larger training batches.",
    )
    parser.add_argument(
        "--batch_train",
        type=int,
        default=20,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--batch_val",
        type=int,
        default=20,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=20,
        help="Input size",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=64,
        help="Hidden size",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for early stopper.",
    )
    parser.add_argument(
        "--pred_horizon",
        type=int,
        choices=[30, 45, 60, 120],
        help="Prediction horizon."
    )
    parser.add_argument(
        "--date_experiment",
        type=str,
        default=datetime.today().strftime('%Y-%m-%d'),
        help="Date of the experiemnt in format YYYY-MM-DD."
    )
    
    args = parser.parse_args()
    
    # config the runname here and overwrite.
    if args.neuro_mapping:
        run_name = f"s_MLP_{args.student_model}_t_simglucose_data_insilico_seed_{args.seed}_{args.date_experiment}_PH_{str(args.pred_horizon)}"
    else:
        run_name = f"s_MLP_{args.student_model}_data_insilico_seed_{args.seed}_{args.date_experiment}_PH_{str(args.pred_horizon)}"
    args.run_name = run_name
    args.dump_path = os.path.join("results","MLP_"+args.student_model)
    if args.modified:
        args.dump_path = os.path.join(args.dump_path,"no_cycles" )
    args.dump_path = os.path.join(args.dump_path, args.run_name)
    trainer = prepare_trainer(args)
    try:
        if args.date_experiment == datetime.today().strftime('%Y-%m-%d'):
            logger.info("Start training.")
            trainer.train()
        else:
            pass
    except Exception as e:
        # Save the training loss values
        with open(os.path.join(trainer.dump_path,'train_loss.pkl'), 'wb') as file:
            dump(trainer.track_loss, file)
        
        # Save the II loss values
        if trainer.neuro_mapping:
            with open(os.path.join(trainer.dump_path,'ii_loss.pkl'), 'wb') as file:
                dump(trainer.track_II_loss, file)
        logger.error(f"Something went wrong :( --> {e}")
    finally:
        logger.info("Start evaluation.")
        trainer.evaluate()
        trainer.test()
