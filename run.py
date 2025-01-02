from lib.config import cfg, args


def to_cuda(batch):
    if isinstance(batch, dict):
        for k in batch:
            if k == 'meta' or k == 'obj':
                continue
            elif isinstance(batch[k], tuple) or isinstance(batch[k], list): 
                batch[k] = [to_cuda(b) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = to_cuda(batch[k])
            else:
                batch[k] = batch[k].cuda()
        return batch
    else:
        return batch.cuda()


def run_tree_infer():
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import torch
    from tqdm import tqdm
    from lib.visualizers import make_visualizer
    from lib.utils.tree import TreeFolder, get_op
    import os.path as osp
    @torch.no_grad()
    def sample_model(model, device, batch, size, temperature=1.0, condition=None):
        row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
        cache = {}
        for i in tqdm(range(size[0])):
            for j in range(size[1]):
                out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
                prob = torch.softmax(out[:, :, i, j] / temperature, 1)
                sample = torch.multinomial(prob, 1).squeeze(-1)
                row[:, i, j] = sample
        return row

    network = make_network(cfg.sample.bottom_network).cuda()
    load_network(network,
                 cfg.sample.bottom_network.path,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch,
                 strict=False)
    network.eval()

    network_top = make_network(cfg.sample.top_network).cuda()
    load_network(
        network_top,
        cfg.sample.top_network.path,
        resume=True,
        epoch=-1,
        strict=False
    )

    sibling_network_top = make_network(cfg.sample.sibling_top_network).cuda()
    load_network(
        sibling_network_top,
        cfg.sample.sibling_top_network.path,
        resume=True,
        epoch=-1,
        strict=False
    )
    sibling_network_top.eval()

    sibling_network_bottom = make_network(cfg.sample.sibling_bottom_network).cuda()
    load_network(
        sibling_network_bottom,
        cfg.sample.sibling_bottom_network.path,
        resume=True,
        epoch=-1,
        strict=False
    )
    sibling_network_bottom.eval()

    vqvae_net = make_network(cfg.sample.vqvae_network).cuda()
    load_network(
        vqvae_net,
        cfg.sample.vqvae_network.path,
        resume=True,
        epoch=-1,
        strict=False
    )
    vqvae_net.eval()

    classify_net = make_network(cfg.sample.classify_network).cuda()
    load_network(
        classify_net,
        cfg.sample.classify_network.path,
        resume=True,
        epoch=-1,
        strict=False
    )
    classify_net.eval()

    ops_net = make_network(cfg.sample.ops_network).cuda()
    load_network(
        ops_net,
        cfg.sample.ops_network.path,
        resume=True,
        epoch=-1,
        strict=False
    )
    ops_net.eval()

    opparam_net = make_network(cfg.sample.opparam_network).cuda()
    load_network(
        opparam_net,
        cfg.sample.opparam_network.path,
        resume=True,
        epoch=-1,
        strict=False
    )
    opparam_net.eval()

    visualizer = make_visualizer(cfg, name=cfg.vis_name)
    result_path = osp.join(cfg.result_dir, visualizer.name, 'tree')
    import json
    test_img_path = [cfg.demo_path]

    tree = TreeFolder(result_path, test_img_path)

    import cv2
    imask = (cv2.imread('./mask.png'))[..., 0]
    imask = cv2.resize(imask, (256, 256))
    imask = imask > 0

    # temporariliy enable all failed nodes
    N = len(tree.nodes)
    for i in range(N):
        if tree.nodes[i].type == -100:
            tree.nodes[i].type = 0
    # tree.nodes[130].type = 0
    import json
    if osp.exists(osp.join(result_path, 'debug.json')):
        with open(osp.join(result_path, 'debug.json'), 'r') as f:
            debug_info = json.load(f)
    else:
        debug_info = [{} for _  in range(10000)]

    repeat = cfg.ti_rp
    assert cfg.ti_bs % repeat == 0
    max_cnt = cfg.ti_bs // repeat

    with torch.no_grad():
        while True:
            # first, in the folder, find all node with unknown type and predict the type using pretrained network
            while True:
                input_imgs, input_ids = tree.get_images_with_unknown_type()
                if len(input_imgs) == 0:
                    break
                input_imgs = torch.stack(input_imgs).cuda()
                pred_prob = classify_net(input_imgs, {})['prob']
                pred_label = torch.argmax(pred_prob, 1)
                tree.set_images_with_unknown_type(input_ids, pred_label)
            # Then, fetch a batch of images and predict child
            input_imgs, input_ids = tree.get_images_with_unknown_child(max_cnt=max_cnt)
            if len(input_imgs) == 0:
                break
            input_imgs = torch.stack(input_imgs).cuda()
            input_imgs[:, :, ~imask] = -1.0
            B = input_imgs.shape[0]
            input_imgs_orig = input_imgs.clone()
            input_imgs = input_imgs.repeat(repeat, 1, 1, 1) # need to be reshaped to (repeat, batch, 3, 64, 64)
            N = input_imgs.shape[0]
            # Encode using vqvae
            vqvae_output = vqvae_net(input_imgs, {})
            latent_top = vqvae_output['latent_t']
            latent_bottom = vqvae_output['latent_b']
            # sample
            top_sample = sample_model(
                network_top.net, 'cuda',  N, [8, 8], condition=[latent_top, latent_bottom]
            )
            sample = sample_model(
                network.net, 'cuda', N, [32, 32], condition=[latent_top, latent_bottom, top_sample]
            )
            img_decoded = vqvae_net.decode_code(top_sample, sample) # (N, 3, H, W)
            sibiling_top_sample = sample_model(
                sibling_network_top.net, 'cuda', N, [8, 8], condition=[latent_top, latent_bottom, top_sample, sample]
            )
            sibling_bottom_sample = sample_model(
                sibling_network_bottom.net, 'cuda', N, [32, 32], condition=[latent_top, latent_bottom, top_sample, sample, sibiling_top_sample]
            )
            # print(sample)
            sibling_img_decoded = vqvae_net.decode_code(sibiling_top_sample, sibling_bottom_sample) # (N, 3, H, W)
            # predict ops

            mse_thresh = 20
            mse_fn = lambda x, y: (((x-y)**2).sum((-1, -2, -3))).sqrt()

            # mapping
            components = img_decoded[:, None].expand(-1, 3, -1, -1, -1)
            mapping_batch = {
                'img': input_imgs,
                'components': components,
            }
            pred_ops = ops_net(input_imgs, mapping_batch)['prob']
            pred_ops_label = pred_ops.argmax(dim=1)
            lchilds = img_decoded
            rchilds = sibling_img_decoded
            all_params = lchilds.clone()
            all_error = torch.zeros(lchilds.shape[0],).cuda()
            convert = lambda x: (x + 1.0) * 0.5
            for i, op in enumerate(cfg.ops):
                mask = pred_ops_label == i
                if mask.sum() == 0:
                    continue
                img_msk = input_imgs[mask]
                lchild_msk = lchilds[mask]
                rchild_msk = rchilds[mask]
                op_func = get_op(i)
                if op in ['fresnel', 'mix']: # need to predicted param
                    components = torch.concat(
                        [
                            lchild_msk[:, None],
                            rchild_msk[:, None],
                            lchild_msk[:, None]
                        ],
                        dim=1
                    )
                    pred_batch = {
                        'components': components,
                    }
                    ret = opparam_net(img_msk, pred_batch)['param_pred']
                    mask_not_good_mask = ret[:, :, imask].mean(dim=-1) > 0.92
                    ret[mask_not_good_mask] = -ret[mask_not_good_mask]
                    all_params[mask] = ret
                    recon = op_func(convert(lchild_msk), convert(rchild_msk), convert(ret))
                else:
                    recon = op_func(convert(lchild_msk), convert(rchild_msk))
                import cv2
                error = mse_fn(recon, convert(img_msk))
                all_error[mask] = error
            lcomp_error = mse_fn(lchilds, input_imgs)
            rcomp_error = mse_fn(rchilds, input_imgs)
            # sim_mse_thresh = 30
            sim_mse_thresh = 5
            lcomp_sim = (lcomp_error < sim_mse_thresh)
            rcomp_sim = (rcomp_error < sim_mse_thresh)
            left_ch = (lchilds + 1.0) * 0.5
            right_ch = (rchilds + 1.0) * 0.5
            vanished_child = (left_ch.reshape(N, -1).sum(-1) < 10) | (right_ch.reshape(N, -1).sum(-1) < 10)
            all_error[lcomp_sim | rcomp_sim | vanished_child] += 100.
            all_error = all_error.reshape(repeat, B)
            idx = all_error.argmin(dim=0)
            aidx = torch.arange(B).cuda()
            error_idx = all_error.min(dim=0)[0]
            lchilds_idx = lchilds.reshape(repeat, B, 3, 256, 256)[idx, aidx]
            rchilds_idx = rchilds.reshape(repeat, B, 3, 256, 256)[idx, aidx]
            params_idx = all_params.reshape(repeat, B, 3, 256, 256)[idx, aidx]
            error_unacceptable = error_idx > mse_thresh
            pred_ops_label_refined = pred_ops_label.clone()
            pred_ops_label_refined = pred_ops_label_refined.reshape(repeat, B)[idx, aidx]
            pred_ops_label = pred_ops_label.reshape(repeat, B)[idx, aidx]
            N = lchilds.shape[0]
            pred_ops_label_refined[error_unacceptable] = 100
            tree.set_images_with_predicted_ops(input_ids, pred_ops_label_refined)
            fresnel_mask = pred_ops_label == 2
            lchilds_idx[:, :, ~imask] = -1.0
            rchilds_idx[:, :, ~imask] = -1.0
            params_idx[:, :, ~imask] = -1.0
            tree.set_images_with_unknown_child(input_ids, lchilds_idx, rchilds_idx, params_idx, fresnel_mask)
            for eid, id in enumerate(input_ids):
                debug_info[id] = {
                    'id': id,
                    'error': error_idx[eid].item(),
                }
            with open(osp.join(result_path, 'debug.json'), 'w') as f:
                json.dump(debug_info, f)
            tree.dump_json()

    tree.dump_json()
    tree.render_graph()

if __name__ == '__main__':
    cfg.split = args.type
    run_tree_infer()
