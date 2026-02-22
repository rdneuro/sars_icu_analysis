#!/usr/bin/env bash
# project_schaefer200.sh
# Project Schaefer-200 (7Networks) from fsaverage → all subjects

SUBJECTS_DIR=/mnt/nvme1n1p1/sars_cov_2_project/data/output/structural/fs_subjects
export SUBJECTS_DIR

ANNOT_NAME="Schaefer2018_200Parcels_7Networks_order"

# 1. Download fsaverage annots (only once)
for hemi in lh rh; do
    if [ ! -f "${SUBJECTS_DIR}/fsaverage/label/${hemi}.${ANNOT_NAME}.annot" ]; then
        wget -O "${SUBJECTS_DIR}/fsaverage/label/${hemi}.${ANNOT_NAME}.annot" \
            "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/${hemi}.${ANNOT_NAME}.annot"
    fi
done

# 2. Project to each subject
for sub_dir in ${SUBJECTS_DIR}/sub-*_T1w; do
    sub=$(basename "${sub_dir}")
    echo "Processing ${sub}..."

    for hemi in lh rh; do
        outfile="${sub_dir}/label/${hemi}.${ANNOT_NAME}.annot"

        if [ -f "${outfile}" ]; then
            echo "  ${hemi} already exists, skipping."
            continue
        fi

        mri_surf2surf --hemi "${hemi}" \
            --srcsubject fsaverage \
            --trgsubject "${sub}" \
            --sval-annot "${SUBJECTS_DIR}/fsaverage/label/${hemi}.${ANNOT_NAME}.annot" \
            --tval "${outfile}"
    done
done

echo "Done. Projected ${ANNOT_NAME} to all subjects."
