"""Alembic env — sync mode (uses sqlalchemy ``+asyncpg`` stripped to plain
postgres URL). Online migrations only — offline mode unsupported."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from pps_embed.schema import Base

config = context.config
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:  # pragma: no cover
        pass

target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("offline migrations not supported; run with online DB URL")
else:
    run_migrations_online()
