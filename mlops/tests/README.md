# Tests

## IMPORTANTE: No modificar los tests para que pasen

Los tests definen el **comportamiento esperado** del sistema. Si un test falla, el código es el que hay que arreglar, no el test.

Está **prohibido**:
- Cambiar asserts para que coincidan con un output incorrecto
- Eliminar tests que fallan
- Añadir `@pytest.mark.skip` para saltarse tests
- Relajar tolerancias sin justificación

Si un test falla legítimamente porque el comportamiento cambió intencionadamente, documenta el cambio en el commit message y pide revisión.
