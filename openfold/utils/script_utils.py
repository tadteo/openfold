import json
import logging
import os
import re
import time
import gc
import psutil

import numpy
import torch

from openfold.model.model import AlphaFold
from openfold.np import residue_constants, protein
from openfold.np.relax import relax
from openfold.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_
)

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def count_models_to_evaluate(openfold_checkpoint_path, jax_param_path):
    model_count = 0
    if openfold_checkpoint_path:
        model_count += len(openfold_checkpoint_path.split(","))
    if jax_param_path:
        model_count += len(jax_param_path.split(","))
    return model_count


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def load_models_from_command_line(config, model_device, openfold_checkpoint_path, jax_param_path, output_dir):
    # Create the output directory

    multiple_model_mode = count_models_to_evaluate(openfold_checkpoint_path, jax_param_path) > 1
    if multiple_model_mode:
        logger.info(f"evaluating multiple models")

    if jax_param_path:
        for path in jax_param_path.split(","):
            model_basename = get_model_basename(path)
            model_version = "_".join(model_basename.split("_")[1:])
            model = AlphaFold(config)
            model = model.eval()
            import_jax_weights_(
                model, path, version=model_version
            )
            model = model.to(model_device)
            logger.info(
                f"Successfully loaded JAX parameters at {path}..."
            )
            output_directory = make_output_directory(output_dir, model_basename, multiple_model_mode)
            yield model, output_directory

    if openfold_checkpoint_path:
        for path in openfold_checkpoint_path.split(","):
            model = AlphaFold(config)
            model = model.eval()
            checkpoint_basename = get_model_basename(path)
            if os.path.isdir(path):
                # A DeepSpeed checkpoint
                ckpt_path = os.path.join(
                    output_dir,
                    checkpoint_basename + ".pt",
                )

                if not os.path.isfile(ckpt_path):
                    convert_zero_checkpoint_to_fp32_state_dict(
                        path,
                        ckpt_path,
                    )
                d = torch.load(ckpt_path)
                import_openfold_weights_(model=model, state_dict=d["ema"]["params"])
            else:
                ckpt_path = path
                d = torch.load(ckpt_path)

                if "ema" in d:
                    # The public weights have had this done to them already
                    d = d["ema"]["params"]
                import_openfold_weights_(model=model, state_dict=d)

            model = model.to(model_device)
            logger.info(
                f"Loaded OpenFold parameters at {path}..."
            )
            output_directory = make_output_directory(output_dir, checkpoint_basename, multiple_model_mode)
            yield model, output_directory

    if not jax_param_path and not openfold_checkpoint_path:
        raise ValueError(
            "At least one of jax_param_path or openfold_checkpoint_path must "
            "be specified."
        )


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split('\W| \|', t)[0] for t in tags]

    return tags, seqs


def update_timings(timing_dict, output_dir):
    """
    Write timing information for a single protein to its own JSON file.
    
    Args:
        timing_dict: Dictionary containing timing info for one protein
        output_dir: Base output directory
    """
    # Create timings directory if it doesn't exist
    timings_dir = os.path.join(output_dir, "timings")
    os.makedirs(timings_dir, exist_ok=True)
    
    # Extract protein tag from the timing dictionary (should only have one key)
    tag = list(timing_dict.keys())[0]
    
    # Create protein-specific timing file
    output_file = os.path.join(timings_dir, f"{tag}_timing.json")
    
    # Load existing timing data for this protein if it exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    
    # Update with new timing information
    timings.update(timing_dict[tag])
    
    # Write to protein-specific file
    with open(output_file, "w") as f:
        json.dump(timings, f, indent=2)
    
    return output_file


def run_model(model, batch, tag, output_dir):
    """
    Run inference using an OpenFold model and track performance metrics.
    
    Args:
        model: The OpenFold model instance
        batch: Input batch containing protein data
        tag: Unique identifier for the protein (e.g. chain ID)
        output_dir: Base output directory for results
        
    Returns:
        out: Model predictions or None if error occurred
        
    Performance metrics (saved to {output_dir}/timings/{tag}_timing.json):
        - Inference time
        - Peak CPU memory usage
        - Peak GPU memory usage (if available)
        - Error status and details if applicable
    """
    try:
        with torch.no_grad():
            # Clear memory before starting
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Temporarily disable templates if there aren't any in the batch
            template_enabled = model.config.template.enabled
            model.config.template.enabled = template_enabled and any([
                "template_" in k for k in batch
            ])

            logger.info(f"Running inference for {tag}...")
            t = time.perf_counter()
            out = model(batch)
            
            # Ensure all GPU operations are completed
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            inference_time = time.perf_counter() - t
            
            # Get peak memory usage
            memory_stats = {
                'cpu_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'status': 'success'
            }
            if torch.cuda.is_available():
                memory_stats['gpu_max_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)

            logger.info(f"Inference time: {inference_time}")
            logger.info(f"Peak CPU Memory: {memory_stats['cpu_memory_mb']:.2f} MB")
            if torch.cuda.is_available():
                logger.info(f"Peak GPU Memory: {memory_stats['gpu_max_memory_mb']:.2f} MB")

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.error(f"Out of memory error for {tag}: {str(e)}")
        memory_stats = {
            'cpu_memory_mb': None,
            'gpu_max_memory_mb': None,
            'status': 'out_of_memory',
            'error': str(e)
        }
        inference_time = None
        out = None
    except Exception as e:
        logger.error(f"Unexpected error for {tag}: {str(e)}")
        memory_stats = {
            'cpu_memory_mb': None,
            'gpu_max_memory_mb': None,
            'status': 'error',
            'error': str(e)
        }
        inference_time = None
        out = None

    # Update timings with protein-specific file
    update_timings({tag: {
        "inference": inference_time,
        "memory": memory_stats
    }}, output_dir)

    model.config.template.enabled = template_enabled

    return out


def prep_output(out, batch, feature_dict, feature_processor, config_preset, multimer_ri_gap, subtract_plddt):
    plddt = out["plddt"]

    plddt_b_factors = numpy.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    if subtract_plddt:
        plddt_b_factors = 100 - plddt_b_factors

    # Prep protein metadata
    template_domain_names = []
    template_chain_index = None
    if feature_processor.config.common.use_templates and "template_domain_names" in feature_dict:
        template_domain_names = [
            t.decode("utf-8") for t in feature_dict["template_domain_names"]
        ]

        # This works because templates are not shuffled during inference
        template_domain_names = template_domain_names[
                                :feature_processor.config.predict.max_templates
                                ]

        if "template_chain_index" in feature_dict:
            template_chain_index = feature_dict["template_chain_index"]
            template_chain_index = template_chain_index[
                                   :feature_processor.config.predict.max_templates
                                   ]

    no_recycling = feature_processor.config.common.max_recycling_iters
    remark = ', '.join([
        f"no_recycling={no_recycling}",
        f"max_templates={feature_processor.config.predict.max_templates}",
        f"config_preset={config_preset}",
    ])

    # For multi-chain FASTAs
    ri = feature_dict["residue_index"]
    chain_index = (ri - numpy.arange(ri.shape[0])) / multimer_ri_gap
    chain_index = chain_index.astype(numpy.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if c != cur_chain:
            cur_chain = c
            prev_chain_max = i + cur_chain * multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=False,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein


def relax_protein(config, model_device, unrelaxed_protein, output_directory, output_name, cif_output=False):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        **config.relax,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    # the struct_str will contain either a PDB-format or a ModelCIF format string
    struct_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein, cif_output=cif_output)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t

    logger.info(f"Relaxation time: {relaxation_time}")
    # Update relaxation timing in protein-specific file
    tag = output_name.split('_')[0]  # Extract protein tag from output name
    update_timings({tag: {
        "relaxation": relaxation_time
    }}, output_directory)

    # Save the relaxed PDB.
    suffix = "_relaxed.pdb"
    if cif_output:
        suffix = "_relaxed.cif"
    relaxed_output_path = os.path.join(
        output_directory, f'{output_name}{suffix}'
    )
    with open(relaxed_output_path, 'w') as fp:
        fp.write(struct_str)

    logger.info(f"Relaxed output written to {relaxed_output_path}...")
