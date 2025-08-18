import argparse

import bittensor as bt
from bittensor.core.config import Config
from bittensor.core.subtensor import Subtensor
from bittensor_wallet.wallet import Wallet


def get_config():
    parser = argparse.ArgumentParser()
    Wallet.add_args(parser=parser)
    Subtensor.add_args(parser=parser)

    config = Config(parser=parser, strict=True)

    return config


def main(netuid: int):
    config = get_config()

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=netuid)
    wallet = Wallet(config=config)

    wallet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    top_64_stake = sorted(metagraph.S)[-64:]

    print(
        f"Current requirement for validator permits based on the top 64 stake stands at {min(top_64_stake)}"
    )

    print(f"Wallet hotkey stake {metagraph.S[wallet_uid]}")

    print(f"Validator permit: {metagraph.validator_permit[wallet_uid]}")


if __name__ == "__main__":
    # Example run command
    # python neurons/validator/scripts/validator.py \
    #     --subtensor.network test \
    #     --wallet.name validator \
    #     --wallet.hotkey default

    main(netuid=155)
