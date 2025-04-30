python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_5000/ --wandb-run-name pong-dqn-5000 --num-episodes 10000 --memory-size 1000000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.05 --target-update-frequency 5000 --replay-start-size 50000 --max-episode-steps 100000 --train-per-step 1 --batch-size 64 --replay-buffer-type prioritized

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_5000_ddqn/ --wandb-run-name pong-dqn-5000-ddqn --num-episodes 10000 --memory-size 1000000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.05 --target-update-frequency 5000 --replay-start-size 50000 --max-episode-steps 100000 --train-per-step 1 --batch-size 64 --replay-buffer-type prioritized --double-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_5000_ddqn_lr_1e-3/ --wandb-run-name pong-dqn-5000-ddqn-lr-1e-3 --num-episodes 10000 --memory-size 1000000 --lr 0.001 --epsilon-decay 0.9999 --epsilon-min 0.05 --target-update-frequency 5000 --replay-start-size 50000 --max-episode-steps 100000 --train-per-step 1 --batch-size 64 --replay-buffer-type prioritized --double-dqn


# Final

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_vanilla/ --wandb-run-name pong-dqn-vanilla --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type uniform --train-frequency 4 --epsilon-decay-steps 1000000

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized/ --wandb-run-name pong-dqn-prioritized --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 1000000

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn/ --wandb-run-name pong-dqn-prioritized-ddqn --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 1000000 --double-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber/ --wandb-run-name pong-dqn-prioritized-ddqn-huber --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 100000 --double-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_v2/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-v2 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v2/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v2 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 25000 --double-dqn --n-step 3 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_5_dueling/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-5-dueling --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 1 --epsilon-decay-steps 25000 --double-dqn --n-step 5 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v3/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v3 --num-episodes 10000 --memory-size 25000 --lr 0.00025 --epsilon-min 0.1 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 1 --epsilon-decay-steps 25000 --double-dqn --n-step 3 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_5_dueling_v2/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-5-dueling-v2 --num-episodes 10000 --memory-size 25000 --lr 0.00025 --epsilon-min 0.1 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 1 --batch-size 32 --replay-buffer-type prioritized --train-frequency 1 --epsilon-decay-steps 25000 --double-dqn --n-step 5 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 \
  --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v4/ \
  --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v4  \
  --num-episodes 10000 \
  --memory-size 50000 \
  --lr 0.0005 \
  --epsilon-start 1.0 \
  --epsilon-min 0.05 \
  --epsilon-decay-steps 50000 \
  --target-update-frequency 500 \
  --replay-start-size 10000 \
  --max-episode-steps 10000 \
  --train-per-step 4 \
  --batch-size 64 \
  --replay-buffer-type prioritized \
  --train-frequency 4 \
  --double-dqn \
  --n-step 3 \
  --dueling-dqn \
  --discount-factor 0.99


python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v5/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v5 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_5_dueling_v2/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-5-dueling-v2 --num-episodes 10000 --memory-size 100000 --lr 0.0005 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 10000 --double-dqn --n-step 5 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v6/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v6 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --


python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_5_dueling/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-5-dueling --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 1 --batch-size 256 --replay-buffer-type prioritized --train-frequency 1 --epsilon-decay-steps 25000 --double-dqn --n-step 5 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v7/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v7 \
--num-episodes 10000 \
--memory-size 50000 \
--lr 0.0005 \
--epsilon-start 0.5 \
--epsilon-min 0.01 \
--epsilon-decay-steps 15000 \
--target-update-frequency 300 \
--replay-start-size 1000 \
--max-episode-steps 100000 \
--train-per-step 8 \
--batch-size 128 \
--replay-buffer-type prioritized \
--train-frequency 1 \
--double-dqn \
--n-step 3 \
--dueling-dqn \
--discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v10/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v10 \
--num-episodes 10000 \
--memory-size 50000 \
--lr 0.00025 \
--epsilon-start 1.0 \
--epsilon-min 0.01 \
--epsilon-decay-steps 100000 \
--target-update-frequency 1000 \
--replay-start-size 5000 \
--max-episode-steps 100000 \
--train-per-step 16 \
--batch-size 32 \
--replay-buffer-type prioritized \
--train-frequency 4 \
--double-dqn \
--n-step 3 \
--dueling-dqn \
--discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v11/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v11 \
--num-episodes 10000 \
--memory-size 100000 \
--lr 0.00025 \
--epsilon-start 1.0 \
--epsilon-min 0.01 \
--epsilon-decay-steps 150000 \
--target-update-frequency 1000 \
--replay-start-size 50000 \
--max-episode-steps 100000 \
--train-per-step 16 \
--batch-size 32 \
--replay-buffer-type prioritized \
--train-frequency 4 \
--double-dqn \
--n-step 3 \
--dueling-dqn \
--discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v9/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v9 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_5_dueling_v4 --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v4 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 16 --replay-buffer-type prioritized --train-frequency 1 --epsilon-decay-steps 25000 --double-dqn --n-step 5 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v12/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v12 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v13/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v13 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 16 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_5_dueling_v5 --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-5-dueling-v5 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 8 --batch-size 16 --replay-buffer-type prioritized --train-frequency 1 --epsilon-decay-steps 50000 --double-dqn --n-step 5 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v13 --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v13 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 8 --batch-size 16 --replay-buffer-type prioritized --train-frequency 1 --epsilon-decay-steps 50000 --double-dqn --n-step 5 --dueling-dqn

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v14/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v14 --num-episodes 10000 --memory-size 100000 --lr 0.0005 --epsilon-start 1.0 --epsilon-min 0.01 --epsilon-decay-steps 50000 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 32 --batch-size 16 --replay-buffer-type prioritized --train-frequency 4 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v16/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v16 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 2000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 8 --batch-size 32 --replay-buffer-type prioritized --train-frequency 8 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v17/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v17 --num-episodes 10000 --memory-size 100000 --lr 0.0005 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v18/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v18 --num-episodes 10000 --memory-size 50000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 2000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 8 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v19/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v19 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 2000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 8 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v20/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v20 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 8 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v22/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v22 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 16 --batch-size 16 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 100000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v23/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v23 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 4 --batch-size 16 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 100000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v24/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v24 --num-episodes 10000 --memory-size 100000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 4 --batch-size 16 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 100000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v25/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v25 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 16 --batch-size 32 --replay-buffer-type prioritized --train-frequency 8 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v26/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v26 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.1 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 16 --batch-size 32 --replay-buffer-type prioritized --train-frequency 8 --epsilon-decay-steps 75000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v28/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v28 --num-episodes 10000 --memory-size 100000 --lr 0.00025 --epsilon-decay 0.9999 --epsilon-min 0.05 --target-update-frequency 500 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 32 --batch-size 32 --replay-buffer-type prioritized --train-frequency 16 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v29/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v29 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v30/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v30 --num-episodes 10000 --memory-size 10000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v31/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v31 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.1 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v33/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v33 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v34/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v34 --num-episodes 10000 --memory-size 12500 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.05 --target-update-frequency 50 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 16 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v35/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v35 --num-episodes 10000 --memory-size 12500 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 50 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 16 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_v36/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-v36 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 100 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

# Inherit from v29
python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v1/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v1 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

# Inherit from v33
python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v2/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v2 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 500 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

# Claude
python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v4/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v4 --num-episodes 10000 --memory-size 50000 --lr 0.00025 --target-update-frequency 1000 --replay-start-size 10000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 1 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v5/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v5 --num-episodes 10000 --memory-size 50000 --lr 0.00025 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 1 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

# Inherit from v29
python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v6/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v6 --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v7/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v6 --num-episodes 10000 --memory-size 25000 --lr 0.00025 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v8/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v8 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v9/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v9 --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 1 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v10/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v10 --num-episodes 10000 --memory-size 25000 --lr 0.00025 --target-update-frequency 1000 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 1 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v11/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v11 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v12/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v12 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v13/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v13 --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v14/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v14 --num-episodes 10000 --memory-size 10000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v15/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v15 --num-episodes 10000 --memory-size 25000 --lr 0.00025 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v16/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v16 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v17/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v17 --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 2 --batch-size 64 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v18/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v18 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v19/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v19 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v20/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v20 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v21/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v21 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v22/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v22 --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v23/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v23 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_prioritized_ddqn_huber_n_step_3_dueling_noisy_v24/ --wandb-run-name pong-dqn-prioritized-ddqn-huber-n-step-3-dueling-noisy-v24 --num-episodes 10000 --memory-size 25000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_wo_noisy_net/ --wandb-run-name pong-dqn-wo-noisy-net --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_wo_dueling/ --wandb-run-name pong-dqn-wo-dueling --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --discount-factor 0.99 --noisy-net

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_wo_PER/ --wandb-run-name pong-dqn-wo-PER --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type uniform --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --noisy-net --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_wo_DDQN/ --wandb-run-name pong-dqn-wo-DDQN --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --n-step 3 --dueling-dqn --noisy-net --discount-factor 0.99

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_wo_huber/ --wandb-run-name pong-dqn-wo-huber --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 3 --dueling-dqn --noisy-net --discount-factor 0.99 --mse-loss

python dqn.py --env-name ALE/Pong-v5 --save-dir results_pong_wo_msr/ --wandb-run-name pong-dqn-wo-msr --num-episodes 10000 --memory-size 50000 --lr 0.0001 --target-update-frequency 500 --replay-start-size 5000 --max-episode-steps 100000 --train-per-step 4 --batch-size 32 --replay-buffer-type prioritized --train-frequency 4 --epsilon-decay-steps 50000 --double-dqn --n-step 1 --dueling-dqn --noisy-net --discount-factor 0.99

