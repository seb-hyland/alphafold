use maestro::prelude::*;
use std::ffi::OsStr;

#[maestro::main]
fn main() {
    match arg!("mode").as_str() {
        "predict" => {
            let input_dir = arg!("input_dir");
            let result: Box<[Result<NodeResult<_>, _>]> = parallelize(
                Path::new(input_dir)
                    .read_dir()
                    .expect("input_dir could not be read!")
                    .flatten(),
                |entry| {
                    let path = entry.path();
                    let molecule_name = path
                        .file_stem()
                        .expect("All paths in input_files should have a resolveable filename!")
                        .to_string_lossy();
                    println!("Started workflow {molecule_name:?}");

                    let [af_out_file] = alphafold(&path, &molecule_name)?.into_array();
                    Ok(af_out_file)
                },
            );
            println!("Processes terminated: {result:#?}");
        }
        "align" => {
            let input_pdbs = inputs!("input_pdbs");
            let alignment_dirs = inputs!("alignment_dirs");
            assert!(
                input_pdbs.len() == alignment_dirs.len(),
                "input_files and alignment_dirs must be equal in length"
            );
            let result: Box<[Result<NodeResult<_>, _>]> =
                parallelize(input_pdbs.iter().zip(alignment_dirs), |(base, others)| {
                    let molecule_name = base
                        .file_stem()
                        .expect("All paths in input_files should have a resolveable filename!")
                        .to_string_lossy();
                    println!("Started workflow {molecule_name:?}");

                    let [pymol_out_file] = pymol(base, others, &molecule_name)?.into_array();
                    Ok(pymol_out_file)
                });
            println!("Processes terminated: {result:#?}");
        }
        _ => panic!("mode must be set to `predict` or `align`"),
    };
}

fn pymol(pdb_file: &Path, input_dir: &Path, molecule_name: &str) -> WorkflowResult {
    let input_files: Vec<_> = input_dir
        .read_dir()?
        .filter_map(|p| {
            if let Ok(entry) = p {
                if entry.path().extension() == Some(OsStr::new("pdb")) {
                    Some(entry)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();
    let pymol_script: String = format!(
        r#"
        from pymol import cmd
        cmd.load("{}", "fusion")
        {}
        with open("alignment_rmsds.txt", "w") as f:
        {}
        cmd.quit()
        "#,
        pdb_file.display(),
        input_files
            .iter()
            .map(|entry| format!(
                r#"cmd.load("{}", "{}")"#,
                entry.path().display(),
                entry.file_name().display()
            ))
            .reduce(|mut initial, current| {
                initial.push('\n');
                initial.push_str(&current);
                initial
            })
            .unwrap_or("".to_owned()),
        input_files
            .iter()
            .map(|entry| format!(
                r#"
                rmsd = cmd.align("fusion", "{}")[0]
                f.write(f"RMSD (fusion vs {}): {{rmsd}}\n")
                "#,
                entry.path().display(),
                entry.path().display()
            ))
            .reduce(|mut initial, current| {
                initial.push('\n');
                initial.push_str(&current);
                initial
            })
            .unwrap_or("".to_owned())
    )
    .lines()
    .map(|line| line.trim())
    .fold(String::new(), |mut initial, current| {
        initial.push_str(current);
        initial.push('\n');
        initial
    });
    let output_file = Path::new("alignment_rmsds.txt");

    process! {
        /// Runs PyMOL to align a reference structure against one or more test structures
        name = format!("pymol_{molecule_name}"),
        executor = "direct",
        args = [pymol_script],
        outputs = [output_file],
        script = r#"
            echo "$pymol_script" > pymol_script.py
            pymol -cq pymol_script.py
        "#
    }
}

fn alphafold(input: &Path, molecule_name: &str) -> WorkflowResult {
    let scratch_dir = Path::new(arg!("scratch_dir"));
    let sif_path = Path::new(arg!("sif_path"));
    let db_dir = Path::new(arg!("db_dir"));

    let out_dir = Path::new("out/").join(molecule_name).join("ranked_0.pdb");

    process! {
        /// Runs AlphaFold to predict the structure of a FASTA file
        name = format!("alphafold_{molecule_name}"),
        executor = "slurm",
        inputs = [input, scratch_dir, sif_path, db_dir],
        outputs = [out_dir],
        dependencies = ["!", "apptainer"],
        script = r#"
            apptainer exec --nv \
              -B /arc/project -B /scratch -B /cvmfs \
              --home="$scratch_dir" \
              "$sif_path" \
              python /opt/alphafold/run_alphafold.py \
                --fasta_paths="$input" \
                --output_dir="out/" \
                --data_dir="$db_dir" \
                --db_preset=full_dbs \
                --model_preset=monomer \
                --bfd_database_path="$db_dir"/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
                --mgnify_database_path="$db_dir"/mgnify/mgy_clusters_2022_05.fa \
                --template_mmcif_dir="$db_dir"/pdb_mmcif/mmcif_files \
                --obsolete_pdbs_path="$db_dir"/pdb_mmcif/obsolete.dat \
                --pdb70_database_path="$db_dir"/pdb70/pdb70 \
                --uniref30_database_path="$db_dir"/uniref30/UniRef30_2021_03 \
                --uniref90_database_path="$db_dir"/uniref90/uniref90.fasta \
                --max_template_date=2023-12-31 \
                --use_gpu_relax=True
    "#
    }
}
