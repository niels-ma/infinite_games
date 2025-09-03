#!/bin/bash

alias gaa="git add ."
alias gcm="git commit -m"
alias gacm="git add . && git commit -m"
alias regen_test_wallet="bash .devcontainer/regen_test_wallet.sh"
alias migration="cd ./neurons/validator && alembic revision -m"

function validator_testnet() {
    python neurons/validator.py \
        --netuid 155 \
        --subtensor.network test \
        --wallet.name validator \
        --wallet.hotkey default \
        --logging.debug 2>&1 | tee validator.log
}

function profile_validator_testnet() {
    py-spy record -o validator_profile_out.json --format speedscope --duration 30 \
        -- python neurons/validator.py \
        --netuid 155 \
        --subtensor.network test \
        --wallet.name validator \
        --wallet.hotkey default
}


function miner_testnet() {
    python neurons/miner.py \
        --netuid 155 \
        --subtensor.network test \
        --wallet.name validator \
        --wallet.hotkey default \
        --logging.debug 2>&1 | tee miner.log
}

export PATH="/usr/local/bin:$PATH"

# this is to simplify setting up the git ssh key for local devcontainers
# this is not needed for codespaces
overwrite_ssh_key() {
    mkdir -p ~/.ssh
    echo "$1" > ~/.ssh/id_rsa && chmod 600 ~/.ssh/id_rsa
    git config --global commit.gpgsign true
    git config --global gpg.format ssh
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa
    git config --global user.signingkey "$(ssh-add -L)"
    touch ~/.ssh/allowed_signers
    ssh-add -L > ~/.ssh/allowed_signers
    git config --global gpg.ssh.allowedSignersFile ~/.ssh/allowed_signers
    if [ -n "$2" ]; then
        git config --global user.name "$2"
        git config --global user.email "$2@users.noreply.github.com"
    fi
}
