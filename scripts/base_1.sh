echo "Run name: ${run_name}"
echo "Model name: ${model_nam}e"
echo "Encoder name: ${encoder_name}"
echo "Encoder weight: ${encoder_weight}"
echo "Epochs: ${epochs}"
echo "Batch size: ${batch_size}"
echo "Learning rate: ${learning_rate}"

echo "Starting training at:     $(date)"

bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Starting training for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false" --field "Model;${model_name};false" --field "Encoder;${encoder_name};false" --field "Encoder weight;${encoder_weight};false"

$HOME/road_segmentation/venv/python $HOME/road_segmentation/src/smp.py -m $model_name -en $encoder_name -ew $encoder_weight -e $epochs -bs $batch_size -lr $learning_rate -bp "${TMPDIR}/additional_data"

echo "Finished training at:     $(date)"

# discord notification on finish
bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Finished training for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false" --field "Model;${model_name};false" --field "Encoder;${encoder_name};false" --field "Encoder weight;${encoder_weight};false"

# End the script with exit code 0
exit 0