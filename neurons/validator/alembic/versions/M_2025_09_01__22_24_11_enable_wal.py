"""Enable WAL

Revision ID: c8a61b7b8c8a
Revises: e71c07407574
Create Date: 2025-09-01 22:24:11.675374

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8a61b7b8c8a"
down_revision: Union[str, None] = "e71c07407574"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            PRAGMA journal_mode = WAL
        """
    )
