import torch

# --- Safely handle tensor or float ---
def to_scalar(x):
    return x.item() if torch.is_tensor(x) else x

def train_model(model, loader, device, num_epochs, loss_fn, optimizer, anatomical_map_tensor_exp):
    epoch_total_losses = []
    epoch_primary_losses = []
    epoch_mse_component_losses = []
    epoch_raw_mse_losses = []
    epoch_anatomy_component_losses = []
    epoch_raw_anatomy_losses = []
    epoch_edge_plain_component_losses = []
    epoch_raw_edge_plain_losses = []
    epoch_edge_anatomy_component_losses = []
    epoch_raw_edge_anatomy_losses = []
    epoch_grad_component_losses = []
    epoch_raw_grad_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        total_primary_loss_sum = 0

        total_mse_component = 0
        total_raw_mse = 0
        total_anatomy_component = 0
        total_raw_anatomy = 0
        total_edge_plain_component = 0
        total_raw_edge_plain = 0
        total_edge_anatomy_component = 0
        total_raw_edge_anatomy = 0
        total_grad_component = 0
        total_raw_grad = 0

        for i, (moving, moving_edge, fixed, fixed_edge) in enumerate(loader):
            moving = moving.to(device)
            moving_edge = moving_edge.to(device)
            fixed = fixed.to(device)
            fixed_edge = fixed_edge.to(device)

            anatomical_map_batch = anatomical_map_tensor_exp.repeat(moving.size(0), 1, 1, 1)

            iteration_idx = epoch * len(loader) + i
            if hasattr(loss_fn, "current_iteration"):
                loss_fn.current_iteration = iteration_idx

            moved_image, flow_field = model(moving, fixed, lambda_val=1.0, registration=True)
            moved_edge = model.transformer(moving_edge, flow_field)

            (
                total_loss,
                loss_mse_component, raw_mse, w_mse,
                loss_anatomy_component, raw_anatomy, w_anatomy,
                loss_edge_plain_component, raw_edge_plain, w_edge_plain,
                loss_edge_anatomy_component, raw_edge_anatomy, w_edge_anatomy,
                loss_grad_component, raw_grad, w_grad
            ) = loss_fn.loss(
                y_true=fixed,
                y_pred=moved_image,
                anatomical_map=anatomical_map_batch,
                flow=flow_field,
                y_true_edge=fixed_edge,
                y_pred_edge=moved_edge
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_epoch_loss += to_scalar(total_loss)
            total_primary = to_scalar(loss_mse_component + loss_anatomy_component +
                                      loss_edge_plain_component + loss_edge_anatomy_component)
            total_primary_loss_sum += total_primary

            total_mse_component += to_scalar(loss_mse_component)
            total_raw_mse += to_scalar(raw_mse)
            total_anatomy_component += to_scalar(loss_anatomy_component)
            total_raw_anatomy += to_scalar(raw_anatomy)
            total_edge_plain_component += to_scalar(loss_edge_plain_component)
            total_raw_edge_plain += to_scalar(raw_edge_plain)
            total_edge_anatomy_component += to_scalar(loss_edge_anatomy_component)
            total_raw_edge_anatomy += to_scalar(raw_edge_anatomy)
            total_grad_component += to_scalar(loss_grad_component)
            total_raw_grad += to_scalar(raw_grad)

        # Save accumulated logs
        epoch_total_losses.append(total_epoch_loss)
        epoch_primary_losses.append(total_primary_loss_sum)
        epoch_mse_component_losses.append(total_mse_component)
        epoch_raw_mse_losses.append(total_raw_mse)
        epoch_anatomy_component_losses.append(total_anatomy_component)
        epoch_raw_anatomy_losses.append(total_raw_anatomy)
        epoch_edge_plain_component_losses.append(total_edge_plain_component)
        epoch_raw_edge_plain_losses.append(total_raw_edge_plain)
        epoch_edge_anatomy_component_losses.append(total_edge_anatomy_component)
        epoch_raw_edge_anatomy_losses.append(total_raw_edge_anatomy)
        epoch_grad_component_losses.append(total_grad_component)
        epoch_raw_grad_losses.append(total_raw_grad)

        # Console output
        print(f"[Epoch {epoch+1}]")
        print(f"  Total Loss (Sum)                : {total_epoch_loss:.4f}")
        print(f"    1. Total Primary Loss (Sum)   : {total_primary_loss_sum:.4f}")
        print(f"      - w_mse * MSE Sum           : {total_mse_component:.4f} (w={w_mse:.3f}, MSE={total_raw_mse:.4f})")
        print(f"      - w_anatomy * AnatomyMSE Sum: {total_anatomy_component:.4f} (w={w_anatomy:.3f}, AnatomyMSE={total_raw_anatomy:.4f})")
        print(f"      - w_edge_plain * EdgePlain  : {total_edge_plain_component:.4f} (w={w_edge_plain:.3f}, EdgeMSE={total_raw_edge_plain:.4f})")
        print(f"      - w_edge_anatomy * EdgeAnatomy: {total_edge_anatomy_component:.4f} (w={w_edge_anatomy:.3f}, EdgeMSE={total_raw_edge_anatomy:.4f})")
        print(f"    2. Grad Loss (Sum)            : {total_grad_component:.4f} (w={w_grad:.3f}, Grad={total_raw_grad:.4f})")

    return {
        'epoch_total_losses': epoch_total_losses,
        'epoch_primary_losses': epoch_primary_losses,
        'epoch_mse_component_losses': epoch_mse_component_losses,
        'epoch_raw_mse_losses': epoch_raw_mse_losses,
        'epoch_anatomy_component_losses': epoch_anatomy_component_losses,
        'epoch_raw_anatomy_losses': epoch_raw_anatomy_losses,
        'epoch_edge_plain_component_losses': epoch_edge_plain_component_losses,
        'epoch_raw_edge_plain_losses': epoch_raw_edge_plain_losses,
        'epoch_edge_anatomy_component_losses': epoch_edge_anatomy_component_losses,
        'epoch_raw_edge_anatomy_losses': epoch_raw_edge_anatomy_losses,
        'epoch_grad_component_losses': epoch_grad_component_losses,
        'epoch_raw_grad_losses': epoch_raw_grad_losses
    }