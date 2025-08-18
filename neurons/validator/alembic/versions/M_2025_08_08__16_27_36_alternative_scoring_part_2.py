"""Alternative scoring part 2

Revision ID: 00ad2c49ac31
Revises: c6848f2e0e3f
Create Date: 2025-08-08 16:27:36.258281

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "00ad2c49ac31"
down_revision: Union[str, None] = "c6848f2e0e3f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE scores ADD COLUMN alternative_processed BOOLEAN NOT NULL DEFAULT false
        """
    )
