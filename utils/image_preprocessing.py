"""Transformações de imagem compartilhadas por treino e inferência."""

from __future__ import annotations

from torchvision import transforms

from config.settings import ImageConfig


def build_image_transform(
    img_config: ImageConfig,
    *,
    augment: bool = False,
) -> transforms.Compose:
    """Cria o pipeline RGB ou grayscale definido pelo contrato do checkpoint.

    ``Grayscale`` produz uma imagem PIL ``L`` de 8 bits (0–255). ``ToTensor``
    converte esses valores para float em [0, 1] e ``Normalize`` aplica a
    padronização configurada antes de a imagem entrar na rede.
    """

    operations: list = [transforms.Resize((img_config.height, img_config.width))]
    if img_config.color_mode == 'grayscale':
        operations.append(transforms.Grayscale(num_output_channels=1))

    if augment:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2 if img_config.color_mode == 'rgb' else 0.0,
            ),
        ])

    operations.append(transforms.ToTensor())
    if img_config.normalize:
        operations.append(transforms.Normalize(
            mean=list(img_config.mean),
            std=list(img_config.std),
        ))
    return transforms.Compose(operations)

