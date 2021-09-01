#Shamelessly ripped from MIRISIM -JB
import logging

from astropy.modeling.functional_models import AiryDisk2D
import scipy.interpolate
import numpy as np
import logging

LOG=logging.getLogger(__name__)


def get_mec_psf(fov, sampling, radius):
    """

    :param fov: (angular extent, angular extent)
    :param sampling: (angular sampling)
    :param radius: (angular radius)
    :return: 2d airy disk placed at center of extent

    for eventual integration with mirisim:
    must have .data.shape[0])) - (.meta.crpix1,2,3 - 1)) * .meta.cdelt1,2,3 + .meta.crval1,2,3
    """

    grid = np.meshgrid(np.linspace(-fov[1] / 2, fov[1] / 2, num=round((fov[1]/sampling).value)),
                       np.linspace(-fov[0] / 2, fov[0] / 2, num=round((fov[0]/sampling).value)))
    sampled_psf = AiryDisk2D(radius=radius)(*grid)
    sampled_psf /= sampled_psf.sum()
    return grid, sampled_psf


def GaussianKernel(x, xcenter, sigma):
    """Return Gaussian with specified xcenter and sigma, evaluated at x."""
    return np.exp(-np.power(x-xcenter, 2.) / (2.*np.power(sigma, 2.)))


def get_points_from_scene(scene, skycube, xy_radec_transform, simulator_config=None):
    """
    Function that returns the combined SkySim outputs for the emission of all
    point sources in a specified scene, for specified skycube and (x,y) to
    (RA,Dec) transformation.

    Parameters
    ----------
    scene : mirisim.skysim.scenes.SkyScene object
        Scene object produced by SkySim, representing the sky illumination as
        evaluated for a grid of positions and wavelengths.
    skycube : MrsSkyCube object
        Skycube object to be populated with extended emission from scene.
    xy_radec_transform : gwcs.wcs.WCS object
        gwcs.wcs.WCS object that represents the transformations between MRS
        Skycube (x,y), MRS channel/band field-of-view (alpha, beta), Focal
        plane (v2,v3), and SkySim (RA, Dec).
    simulator_config : mirisim.config_parser.SimulatorConfig object, optional
        The configuration object that describes the simulator.
    """

    LOG.debug("Evaluating point sources in scene.")

    # use standard installed simulator config if none provided
    if simulator_config is None:
        from mirisim.config_parser import SimulatorConfig
        simulator_config = SimulatorConfig.from_default()

    # Initialize cube containing the point source emission
    points = np.zeros(skycube.data.shape)

    # Assume that the MrsSkyCube has the shape (nlambda, nalpha, nbeta),
    # swap axes on the working grid to work with (nlambda, nbeta, nalpha).
    points = points.swapaxes(1, 2)
    n_al_for_sc = points.shape[2]
    n_be_for_sc = points.shape[1]

    # Load PSF CDP.
    _, dm = get_mec_psf()


    normalize_psf = True

    # Compute alpha, beta, lambda grids of PSF.
    # Assume the PSF CDP data cube has shape (nlambda, nbeta, nalpha).
    psf_al = (np.array(np.arange(dm.data.shape[2])) - (dm.meta.crpix1 - 1)) * dm.meta.cdelt1 + dm.meta.crval1
    psf_be = (np.array(np.arange(dm.data.shape[1])) - (dm.meta.crpix2 - 1)) * dm.meta.cdelt2 + dm.meta.crval2
    psf_wl = (np.array(np.arange(dm.data.shape[0])) - (dm.meta.crpix3 - 1)) * dm.meta.cdelt3 + dm.meta.crval3

    # Assume the PSF alpha, beta grids are symmetric:
    # Get alpha and beta half-width of PSF.
    psf_al_hw = psf_al[-1]
    psf_be_hw = psf_be[-1]

    # Create interpolated PSF cube at requested wavelengths.
    LOG.debug("Interpolating MRS PSF CDP at requested wavelength grid; "
              "clipping outlier wavelengths to the edges of the defined "
              "wavelength grid.")

    # Determine fractional pixel indices in PSF cube that corresponds to
    # requested wavelength grid.
    wl_ind_for_sample = (skycube.lambda_center - dm.meta.crval3) / dm.meta.cdelt3 + dm.meta.crpix3 - 1

    # Clip outlier wavelengths to nearest edge.
    wl_ind_for_sample.clip(0, psf_wl.size - 1, out=wl_ind_for_sample)

    # Create a sampled PSF grid at the requested wavelengths.
    coords = np.meshgrid(wl_ind_for_sample, np.arange(psf_be.size),
                         np.arange(psf_al.size), indexing='ij')
    wl_sampled_psf = scipy.ndimage.interpolation.map_coordinates(
        dm.data.astype(np.float64), coords, prefilter=False, order=1)
    del coords

    # If necessary, normalize the sampled PSF within each wavelength slice.
    if normalize_psf:
        LOG.debug("PSF CDP is from before CDP-7, normalizing in each wavelength slice.")
        wl_sampled_psf = np.swapaxes(
            np.swapaxes(wl_sampled_psf, 0, 2) / np.sum(wl_sampled_psf, axis=(1, 2)), 2, 0)

    # Assume skycube alpha and beta grids are linearily spaced.
    # Determine alpha, beta grid at resolution of Skycube that spans the
    # range in alpha, beta for which the PSF is defined.
    npix_for_al_hw = int(np.ceil(psf_al_hw / skycube.alpha_delta[0]))
    al_hw = npix_for_al_hw * skycube.alpha_delta[0]
    n_al_for_psf = 2 * npix_for_al_hw + 1

    npix_for_be_hw = int(np.ceil(psf_be_hw / skycube.beta_delta[0]))
    be_hw = npix_for_be_hw * skycube.beta_delta[0]
    n_be_for_psf = 2 * npix_for_be_hw + 1

    al_for_psf = np.linspace(-al_hw, al_hw, n_al_for_psf)
    be_for_psf = np.linspace(-be_hw, be_hw, n_be_for_psf)

    # Get a list of point sources
    sources = scene.retrieve_sources('point')

    # For each point source, add its emission to cube
    for ind_source, source in enumerate(sources):

        # Get center of source in SkySim RA, DEC frame
        skys_ra, skys_dec = source.Cen

        # Convert center of source from SkySim (RA,DEC) to (alpha,beta)
        al_source, be_source = xy_radec_transform.get_transform(
            'skysim_radec', 'mrsfov_ab')(skys_ra, skys_dec)

        # Determine if PSF at source position would have overlap with the
        # skycube FoV; if not, then continue with next source.
        if (al_source > skycube.alpha_edge[-1] + psf_al_hw
                or al_source < skycube.alpha_edge[0] - psf_al_hw
                or be_source > skycube.beta_edge[-1] + psf_be_hw
                or be_source < skycube.beta_edge[0] - psf_be_hw):
            LOG.debug("Skipping point source {} (name: \"{}\"): no overlap PSF with skycube "
                      "FoV".format(ind_source, source.name))
            continue

        # Assume skycube alpha and beta grids are linearly spaced.
        # Determine for current point source the fractional pixel offset
        # from center of lower-left pixel on the skycube alpha, beta grid;
        # this can fall outside the skycube FoV.
        al_pixoff = (al_source-skycube.alpha_center[0])/skycube.alpha_delta[0]
        be_pixoff = (be_source-skycube.beta_center[0])/skycube.beta_delta[0]

        # Determine nearest grid point and fractional offset from that point.
        al_pix_near = int(np.round(al_pixoff))
        be_pix_near = int(np.round(be_pixoff))
        al_pix_froff = al_pixoff - al_pix_near
        be_pix_froff = be_pixoff - be_pix_near

        # Compute fraction pixel offset in alpha, beta
        al_froff = al_pix_froff * skycube.alpha_delta[0]
        be_froff = be_pix_froff * skycube.beta_delta[0]

        # Compute a grid with the fractional alpha, beta offset applied.
        al_for_psf_with_froff = al_for_psf - al_froff
        be_for_psf_with_froff = be_for_psf - be_froff

        # Compute of pixel edges with fractional alpha, beta offset applied.
        d_al_for_psf = skycube.alpha_delta[0]
        al_for_psf_with_froff -= d_al_for_psf / 2
        al_for_psf_with_froff = np.concatenate((al_for_psf_with_froff,
                                                al_for_psf_with_froff[-1:] + d_al_for_psf))

        d_be_for_psf = skycube.beta_delta[0]
        be_for_psf_with_froff -= d_be_for_psf / 2
        be_for_psf_with_froff = np.concatenate((be_for_psf_with_froff,
                                                be_for_psf_with_froff[-1:] + d_be_for_psf))

        # Down-sample the "wavelength-sampled" PSF cube to the resolution of
        # skycube, and interpolate at the fractional offset.
        al_ind_for_sample = (al_for_psf_with_froff - dm.meta.crval1) / dm.meta.cdelt1 + dm.meta.crpix1 - 1
        be_ind_for_sample = (be_for_psf_with_froff - dm.meta.crval2) / dm.meta.cdelt2 + dm.meta.crpix2 - 1

        # Among grid of (alpha, beta) coordinates for which to sample the PSF,
        # identify which do not fall on existing points in the PSF CDP
        # coordinate grid. Get interpolated values from PSF CDP for these
        # points.
        al_toadd = (al_ind_for_sample - al_ind_for_sample.astype(int)) != 0
        be_toadd = (be_ind_for_sample - be_ind_for_sample.astype(int)) != 0

        # Create working alpha and beta grid spanning all pixel points within
        # minimum and maximum alpha/beta values that are being sampled.
        # Note: these grids may include points that fall outside the grid
        # for which the original PSF is defined. For these grid points, the
        # interpolated values will be zero, so they will not contribute.
        al_ind_exist = np.arange(int(np.ceil(al_ind_for_sample.min())), int(al_ind_for_sample.max() + 1))
        be_ind_exist = np.arange(int(np.ceil(be_ind_for_sample.min())), int(be_ind_for_sample.max() + 1))

        # Create a union of the working alpha and beta grids and the grid of
        # points for which interpolation was needed.
        al_union = np.concatenate([al_ind_exist, al_ind_for_sample[al_toadd]])
        be_union = np.concatenate([be_ind_exist, be_ind_for_sample[be_toadd]])
        al_union.sort()
        be_union.sort()

        # Get samples for new rows and columns separately.
        coords = np.meshgrid(np.arange(wl_ind_for_sample.size), be_ind_for_sample[be_toadd],
                             al_union, indexing='ij')
        cubenewbe = scipy.ndimage.interpolation.map_coordinates(wl_sampled_psf, coords, prefilter=False, order=1)

        coords = np.meshgrid(np.arange(wl_ind_for_sample.size), be_ind_exist,
                             al_ind_for_sample[al_toadd], indexing='ij')
        cubenewal = scipy.ndimage.interpolation.map_coordinates(wl_sampled_psf, coords, prefilter=False, order=1)
        del coords

        # Create working cube of sampled PSF, based on union of grid from
        # original PSF and the grid for which sampling is needed.
        cube = np.zeros((wl_ind_for_sample.size, be_union.size, al_union.size))

        # Create mesh grids where interpolated PSF points should be put in the working
        # cube, and use these to set the interpolated PSF values in the working cube.
        al_new_ind_be = np.arange(al_union.size)
        be_new_ind_be = np.searchsorted(be_union, be_ind_for_sample[be_toadd])
        be_mesh_out, al_mesh_out = np.meshgrid(be_new_ind_be, al_new_ind_be, indexing='ij')
        cube[:, be_mesh_out, al_mesh_out] = cubenewbe

        al_new_ind_al = np.searchsorted(al_union, al_ind_for_sample[al_toadd])
        be_new_ind_al = np.searchsorted(be_union, be_ind_exist)
        be_mesh_out, al_mesh_out = np.meshgrid(be_new_ind_al, al_new_ind_al, indexing='ij')
        cube[:, be_mesh_out, al_mesh_out] = cubenewal

        # Identify which coordinates from original PSF CDP grid should be
        # copied across to working cube; create corresponding mesh grids.
        # Note: any points in working grids that fall outside the original
        # PSF grid will remain at their initial value of 0, which means they
        # will simply not contribute to the PSF.
        be_new_ind = np.searchsorted(be_union, be_ind_for_sample[be_toadd])
        al_new_ind = np.searchsorted(al_union, al_ind_for_sample[al_toadd])
        al_exist_ind_out = [i for i in range(0, al_union.size)
                            if i not in al_new_ind
                            and 0 <= al_union[i] < psf_al.size]
        be_exist_ind_out = [i for i in range(0, be_union.size)
                            if i not in be_new_ind
                            and 0 <= be_union[i] < psf_be.size]
        be_mesh_out, al_mesh_out = np.meshgrid(be_exist_ind_out, al_exist_ind_out, indexing='ij')

        # Identify to which coordinates in the working cube the original PSF
        # CDP values should be copied to; create corresponding mesh grids.
        al_exist_ind_nonzero = al_ind_exist[(al_ind_exist >= 0) & (al_ind_exist < psf_al.size)]
        be_exist_ind_nonzero = be_ind_exist[(be_ind_exist >= 0) & (be_ind_exist < psf_be.size)]
        be_mesh_exist, al_mesh_exist = np.meshgrid(be_exist_ind_nonzero, al_exist_ind_nonzero, indexing='ij')

        # Update working cube by copying across original PSF CDP values for
        # those coordinates that fell on existing original PSF grid.
        cube[:, be_mesh_out, al_mesh_out] = wl_sampled_psf[:, be_mesh_exist, al_mesh_exist]

        # Perform cumulative integration along alpha axis, then take difference
        # between grid points (edges) for which sampling is needed.
        cube = scipy.integrate.cumtrapz(cube, al_union, initial=0, axis=-1)
        cube = np.diff(cube[:, :, np.searchsorted(al_union, al_ind_for_sample)], axis=-1)

        # Perform cumulative integration along beta axis, then take difference
        # between grid points (edges) for which sampling is needed.
        cube = scipy.integrate.cumtrapz(cube, be_union, initial=0, axis=-2)
        cube = np.diff(cube[:, np.searchsorted(be_union, be_ind_for_sample), :], axis=-2)

        spat_sampled_psf = cube

        # Determine overlapping region with SkyCube alpha, beta grid.
        # Indices of overlap in the sampled psf cube
        ind_samppsf_al_center = int(np.floor(n_al_for_psf / 2))
        ind_samppsf_be_center = int(np.floor(n_be_for_psf / 2))
        ind_al_max = skycube.alpha_center.size - al_pix_near
        ind_al_min = -al_pix_near
        ind_be_max = skycube.beta_center.size - be_pix_near
        ind_be_min = -be_pix_near
        al_ind_for_samppsf = (max(ind_samppsf_al_center + ind_al_min, 0),
                              min(ind_samppsf_al_center + ind_al_max, n_al_for_psf))
        be_ind_for_samppsf = (max(ind_samppsf_be_center + ind_be_min, 0),
                              min(ind_samppsf_be_center + ind_be_max, n_be_for_psf))

        # Indices of overlap in skycube FoV
        al_ind_for_sc = (max(al_pix_near - npix_for_al_hw, 0),
                         min(al_pix_near + npix_for_al_hw + 1, n_al_for_sc))
        be_ind_for_sc = (max(be_pix_near - npix_for_be_hw, 0),
                         min(be_pix_near + npix_for_be_hw + 1, n_be_for_sc))

        # Sample the source SED on the wavelength grid of the skycube.
        # Assume that SED() expects the edges of wavelength bins, in units of
        # [micron].
        sed = source.SED(skycube.lambda_edge)
        # Assume SkySim's SED() returns the 1 extra dummy value for backward
        # compatibility, and remove this 1 element to get an array of
        # same size as the grid of centers of wavelength bins.
        sed = sed[:-1]

        # Convolve the source SED (lambda by lambda) with Gaussian profile
        # (with width set to delta lambda at given lambda)
        delta_lambda = skycube.lambda_delta
        sed_conv = np.zeros(len(sed))
        x = range(len(sed))
        for ind_x in x:
            kernel = GaussianKernel(skycube.lambda_center,
                                    skycube.lambda_center[ind_x],
                                    delta_lambda[ind_x])
            sed_conv += sed[ind_x] * kernel / np.sum(kernel)

        # Assume that SkySim returns the SED fluxes as spectral flux densities
        # in units of [microJy]. Convert from [microJy] to [W/m^2/Hz],
        # with Jy = 1E-26 [W/m^2/Hz]
        sed_conv = sed_conv * 1e-32

        # Get telescope area, in units of [m^2]
        telarea = simulator_config['MiriSim']['Telescope']['telescope_area']

        # Convert the SED flux [W/m^2/Hz] to spatially integrated flux [W/Hz]
        # by multiplying by the telescope area.
        sed_conv = sed_conv * telarea

        # Convert the flux from [W/Hz] to [photons/sec].
        #
        # First convert from Fnu to Flambda:
        # F_lambda [W/m] = (c/lambda**2) * F_nu [W/Hz].
        #
        # Then spectrally integrate, multiplying by d_lambda
        # F [W] = F_lambda [W/m] * d_lambda
        #
        # Then divide by energy of photons to get flux in [photons/s]
        # E_photon = h * c / lambda
        # F [photons/s] = F [W] / E_photon [J]
        #
        # Combine and reduce formulas:
        # F [photons/s] = (c/lambda**2) * F_nu [W/Hz] * d_lambda / (h*c/lambda)
        # F [photons/s] = F_nu [W/Hz] * d_lambda / (lambda * h)
        # F [photons/s] = F_nu [W/Hz] / (R * h)
        # with spectral resolution: R = lambda/d_lambda
        #
        # Importing Planck constant from SciPy, which is in units of
        # [Joule second].
        sed_conv = sed_conv * skycube.lambda_delta / (skycube.lambda_center * scipy.constants.h)

        # Multiply the sampled PSF with the SED. Swapping axes in the sampled
        # PSF to move the wavelength axis to become the first axis, before
        # multiplying by the SED; then swapping axes back again.
        spat_sampled_psf_scaled = (
            spat_sampled_psf.swapaxes(0, 2) * sed_conv).swapaxes(0, 2)

        # Add the overlapping regions of the scaled spatially sampled PSF to
        # skycube FoV cube.
        points[:, be_ind_for_sc[0]:be_ind_for_sc[1], al_ind_for_sc[0]:al_ind_for_sc[1]] += \
            spat_sampled_psf_scaled[:, be_ind_for_samppsf[0]:be_ind_for_samppsf[1], al_ind_for_samppsf[0]:al_ind_for_samppsf[1]]

    # Swap axes on the working grid of (nlambda, nbeta, nalpha), to match
    # the shape of the MrsSkycube (nlambda, nalpha, nbeta).
    points = points.swapaxes(1, 2)

    return points
