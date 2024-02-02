import copy
import utils
import argparse
import wandb
from build_af_dataset import AFDataLoader,split_datasets

af_300 = {
    'name': 'af_300',
    'atom_encoder': {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11,
                     'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19},
    'atom_decoder': ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"],
    'max_n_nodes': 300,
    'n_nodes': {16: 1, 17: 1, 18: 1, 20: 3, 23: 1, 26: 2, 34: 1, 37: 1, 38: 1, 44: 1, 45: 1, 47: 1, 52: 1, 54: 1,
                55: 1, 61: 2, 63: 3, 64: 2, 67: 2, 69: 1, 71: 3, 72: 2, 74: 1, 75: 2, 76: 1, 78: 1, 80: 1, 82: 2,
                85: 1, 86: 1, 91: 4, 92: 1, 93: 2, 94: 1, 96: 2, 97: 1, 99: 1, 100: 4, 101: 4, 103: 4, 104: 4,
                105: 1, 106: 2, 107: 2, 108: 5, 109: 4, 110: 1, 111: 4, 112: 4, 113: 3, 114: 4, 115: 3, 116: 3,
                117: 6, 118: 7, 120: 4, 121: 8, 122: 6, 123: 2, 124: 2, 125: 8, 126: 8, 127: 4, 128: 4, 129: 8,
                130: 10, 131: 4, 132: 6, 133: 3, 134: 6, 135: 5, 136: 8, 137: 8, 138: 7, 139: 7, 140: 5, 141: 7,
                142: 6, 143: 8, 144: 9, 145: 7, 146: 9, 147: 11, 148: 10, 149: 8, 150: 13, 151: 5, 152: 9, 153: 6,
                154: 5, 155: 3, 156: 13, 157: 15, 158: 6, 159: 14, 160: 16, 161: 13, 162: 5, 163: 6, 164: 11, 165: 7,
                166: 9, 167: 7, 168: 10, 169: 12, 170: 7, 171: 9, 172: 14, 173: 8, 174: 3, 175: 9, 176: 6, 177: 7,
                178: 9, 179: 7, 180: 11, 181: 8, 182: 9, 183: 11, 184: 19, 185: 10, 186: 9, 187: 8, 188: 5, 189: 10,
                190: 13, 191: 11, 192: 9, 193: 14, 194: 11, 195: 8, 196: 14, 197: 13, 198: 10, 199: 20, 200: 9,
                201: 12, 202: 16, 203: 13, 204: 8, 205: 11, 206: 7, 207: 13, 208: 10, 209: 14, 210: 9, 211: 13,
                212: 15, 213: 22, 214: 9, 215: 8, 216: 10, 217: 16, 218: 15, 219: 12, 220: 19, 221: 15, 222: 15,
                223: 20, 224: 19, 225: 15, 226: 18, 227: 19, 228: 19, 229: 15, 230: 23, 231: 16, 232: 14, 233: 7,
                234: 9, 235: 19, 236: 20, 237: 14, 238: 20, 239: 15, 240: 14, 241: 11, 242: 16, 243: 17, 244: 9,
                245: 16, 246: 22, 247: 17, 248: 17, 249: 21, 250: 38, 251: 16, 252: 36, 253: 22, 254: 24, 255: 14,
                256: 18, 257: 15, 258: 19, 259: 17, 260: 22, 261: 27, 262: 15, 263: 14, 264: 22, 265: 10, 266: 28,
                267: 31, 268: 13, 269: 14, 270: 23, 271: 17, 272: 13, 273: 18, 274: 25, 275: 24, 276: 37, 277: 13,
                278: 14, 279: 15, 280: 16, 281: 19, 282: 27, 283: 11, 284: 15, 285: 30, 286: 29, 287: 23, 288: 20,
                289: 18, 290: 22, 291: 23, 292: 10, 293: 31, 294: 17, 295: 15, 296: 20, 297: 12, 298: 13, 299: 19,
                300: 22},
    }

af_600 = {
    'name': 'af_600',
    'atom_encoder': {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11,
                     'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19},
    'atom_decoder': ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"],
    'max_n_nodes': 600,
    'n_nodes': {16: 1, 17: 1, 18: 1, 20: 3, 23: 1, 26: 2, 34: 1, 37: 1, 38: 1, 44: 1, 45: 1, 47: 1, 52: 1, 54: 1,
                55: 1, 61: 2, 63: 3, 64: 2, 67: 2, 69: 1, 71: 3, 72: 2, 74: 1, 75: 2, 76: 1, 78: 1, 80: 1, 82: 2,
                85: 1, 86: 1, 91: 4, 92: 1, 93: 2, 94: 1, 96: 2, 97: 1, 99: 1, 100: 4, 101: 4, 103: 4, 104: 4,
                105: 1, 106: 2, 107: 2, 108: 5, 109: 4, 110: 1, 111: 4, 112: 4, 113: 3, 114: 4, 115: 3, 116: 3,
                117: 6, 118: 7, 120: 4, 121: 8, 122: 6, 123: 2, 124: 2, 125: 8, 126: 8, 127: 4, 128: 4, 129: 8,
                130: 10, 131: 4, 132: 6, 133: 3, 134: 6, 135: 5, 136: 8, 137: 8, 138: 7, 139: 7, 140: 5, 141: 7,
                142: 6, 143: 8, 144: 9, 145: 7, 146: 9, 147: 11, 148: 10, 149: 8, 150: 13, 151: 5, 152: 9, 153: 6,
                154: 5, 155: 3, 156: 13, 157: 15, 158: 6, 159: 14, 160: 16, 161: 13, 162: 5, 163: 6, 164: 11, 165: 7,
                166: 9, 167: 7, 168: 10, 169: 12, 170: 7, 171: 9, 172: 14, 173: 8, 174: 3, 175: 9, 176: 6, 177: 7,
                178: 9, 179: 7, 180: 11, 181: 8, 182: 9, 183: 11, 184: 19, 185: 10, 186: 9, 187: 8, 188: 5, 189: 10,
                190: 13, 191: 11, 192: 9, 193: 14, 194: 11, 195: 8, 196: 14, 197: 13, 198: 10, 199: 20, 200: 9,
                201: 12, 202: 16, 203: 13, 204: 8, 205: 11, 206: 7, 207: 13, 208: 10, 209: 14, 210: 9, 211: 13,
                212: 15, 213: 22, 214: 9, 215: 8, 216: 10, 217: 16, 218: 15, 219: 12, 220: 19, 221: 15, 222: 15,
                223: 20, 224: 19, 225: 15, 226: 18, 227: 19, 228: 19, 229: 15, 230: 23, 231: 16, 232: 14, 233: 7,
                234: 9, 235: 19, 236: 20, 237: 14, 238: 20, 239: 15, 240: 14, 241: 11, 242: 16, 243: 17, 244: 9,
                245: 16, 246: 22, 247: 17, 248: 17, 249: 21, 250: 38, 251: 16, 252: 36, 253: 22, 254: 24, 255: 14,
                256: 18, 257: 15, 258: 19, 259: 17, 260: 22, 261: 27, 262: 15, 263: 14, 264: 22, 265: 10, 266: 28,
                267: 31, 268: 13, 269: 14, 270: 23, 271: 17, 272: 13, 273: 18, 274: 25, 275: 24, 276: 37, 277: 13,
                278: 14, 279: 15, 280: 16, 281: 19, 282: 27, 283: 11, 284: 15, 285: 30, 286: 29, 287: 23, 288: 20,
                289: 18, 290: 22, 291: 23, 292: 10, 293: 31, 294: 17, 295: 15, 296: 20, 297: 12, 298: 13, 299: 19,
                300: 22, 301: 16, 302: 14, 303: 21, 304: 16, 305: 17, 306: 18, 307: 26, 308: 17, 309: 23, 310: 25,
                311: 23, 312: 19, 313: 24, 314: 30, 315: 31, 316: 30, 317: 26, 318: 21, 319: 32, 320: 34, 321: 22,
                322: 15, 323: 24, 324: 39, 325: 33, 326: 23, 327: 30, 328: 25, 329: 28, 330: 32, 331: 23, 332: 11,
                333: 21, 334: 20, 335: 25, 336: 26, 337: 25, 338: 43, 339: 23, 340: 20, 341: 16, 342: 23, 343: 18,
                344: 17, 345: 27, 346: 27, 347: 17, 348: 29, 349: 17, 350: 19, 351: 15, 352: 19, 353: 24, 354: 26,
                355: 15, 356: 22, 357: 10, 358: 11, 359: 25, 360: 15, 361: 21, 362: 24, 363: 22, 364: 27, 365: 14,
                366: 18, 367: 23, 368: 17, 369: 17, 370: 9, 371: 20, 372: 14, 373: 19, 374: 17, 375: 20, 376: 14,
                377: 24, 378: 16, 379: 22, 380: 24, 381: 39, 382: 32, 383: 29, 384: 23, 385: 14, 386: 22, 387: 11,
                388: 23, 389: 18, 390: 19, 391: 17, 392: 22, 393: 18, 394: 22, 395: 10, 396: 13, 397: 25, 398: 24,
                399: 17, 400: 11, 401: 13, 402: 10, 403: 22, 404: 11, 405: 18, 406: 15, 407: 16, 408: 16, 409: 16,
                410: 14, 411: 13, 412: 19, 413: 15, 414: 13, 415: 20, 416: 17, 417: 18, 418: 14, 419: 19, 420: 21,
                421: 20, 422: 15, 423: 16, 424: 21, 425: 27, 426: 11, 427: 14, 428: 12, 429: 17, 430: 12, 431: 21,
                432: 16, 433: 18, 434: 23, 435: 14, 436: 14, 437: 22, 438: 15, 439: 11, 440: 16, 441: 16, 442: 15,
                443: 16, 444: 15, 445: 10, 446: 13, 447: 16, 448: 17, 449: 12, 450: 15, 451: 6, 452: 13, 453: 34,
                454: 12, 455: 27, 456: 18, 457: 14, 458: 11, 459: 14, 460: 15, 461: 20, 462: 16, 463: 15, 464: 11,
                465: 14, 466: 19, 467: 22, 468: 13, 469: 17, 470: 12, 471: 22, 472: 11, 473: 21, 474: 26, 475: 16,
                476: 18, 477: 12, 478: 22, 479: 13, 480: 30, 481: 17, 482: 11, 483: 21, 484: 21, 485: 9, 486: 11,
                487: 12, 488: 22, 489: 10, 490: 11, 491: 18, 492: 19, 493: 21, 494: 14, 495: 19, 496: 18, 497: 9,
                498: 12, 499: 16, 500: 32, 501: 20, 502: 15, 503: 16, 504: 12, 505: 19, 506: 19, 507: 12, 508: 14,
                509: 15, 510: 24, 511: 16, 512: 19, 513: 12, 514: 13, 515: 14, 516: 9, 517: 5, 518: 7, 519: 15,
                520: 13, 521: 19, 522: 12, 523: 7, 524: 12, 525: 8, 526: 11, 527: 9, 528: 13, 529: 13, 530: 4, 531: 15,
                532: 13, 533: 16, 534: 7, 535: 17, 536: 18, 537: 7, 538: 15, 539: 11, 540: 10, 541: 16, 542: 8, 543: 11,
                544: 12, 545: 12, 546: 16, 547: 11, 548: 25, 549: 17, 550: 22, 551: 18, 552: 8, 553: 16, 554: 16,
                555: 28, 556: 15, 557: 19, 558: 18, 559: 18, 560: 15, 561: 15, 562: 24, 563: 20, 564: 16, 565: 18,
                566: 7, 567: 11, 568: 13, 569: 20, 570: 9, 571: 9, 572: 7, 573: 10, 574: 19, 575: 9, 576: 15, 577: 4,
                578: 5, 579: 14, 580: 18, 581: 11, 582: 13, 583: 14, 584: 11, 585: 12, 586: 9, 587: 12, 588: 10, 589: 12,
                590: 11, 591: 13, 592: 8, 593: 10, 594: 13, 595: 15, 596: 9, 597: 10, 598: 19, 599: 12},
    }

dataset_info = af_300

from os.path import join
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle

from train_test_af import train_epoch, test
from build_af_dataset import AFDataLoader

parser = argparse.ArgumentParser(description='e3_diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)

parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=5e-5)

parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=192,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)

parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=50)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='multiple arguments can be passed, '
                         'including: homo | onehot | lumo | num_atoms | etc. '
                         'usage: "--conditioning H_thermo homo onehot H_thermo"')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=5000)
parser.add_argument('--normalization_factor', type=float,
                    default=100, help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean" aggregation for the graph network')
args = parser.parse_args()


# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')



# Retrieve dataloaders
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

train_dataset, test_dataset, val_dataset = split_datasets(device)    

dataloaders = {}

dataloaders['train'] = AFDataLoader(dataset=train_dataset, batch_size=args.batch_size)
dataloaders['valid'] = AFDataLoader(dataset=val_dataset, batch_size=args.batch_size)
dataloaders['test'] = AFDataLoader(dataset=test_dataset, batch_size=args.batch_size)

context_node_nf = 0
property_norms = None

args.context_node_nf = context_node_nf


#TODO
# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        #TODO
        # def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            #TODO
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
