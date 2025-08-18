"""Drop deprecated fields in events table

Revision ID: 3063780b8c89
Revises: 057a9fe8f6af
Create Date: 2025-06-19 13:01:50.527562

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3063780b8c89"
down_revision: Union[str, None] = "057a9fe8f6af"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE events DROP COLUMN starts
        """
    )

    op.execute(
        """
            ALTER TABLE events DROP COLUMN resolve_date
        """
    )

    op.execute(
        """
            ALTER TABLE events DROP COLUMN end_date
        """
    )
