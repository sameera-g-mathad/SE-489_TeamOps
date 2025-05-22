from team_ops import Model


def main():
    """
    Main function to train the model.
    """
    # Initialize the model
    model = Model()

    # Get the dataset
    model.load_model()

    # Train the model
    prediction = model.predict(
        """
        We present an independent discovery and detailed characterisation of K2-280b, 
        a transiting low density warm sub-Saturn in a 19.9-day moderately eccentric orbit 
        (e = 0.35_{-0.04}^{+0.05}) from K2 campaign 7. A joint analysis of high precision HARPS,
        HARPS-N, and FIES radial velocity measurements and K2 photometric data indicates that
        K2-280b has a radius of R_b = 7.50 +/- 0.44 R_Earth and a mass of M_b = 37.1 +/- 5.6 M_Earth,
        yielding a mean density of 0.48_{-0.10}^{+0.13} g/cm^3. The host star is a mildly 
        evolved G7 star with an effective temperature of T_{eff} = 5500 +/- 100 K,
        a surface gravity of log(g) = 4.21 +/- 0.05 (cgs), and an iron abundance 
        of [Fe/H] = 0.33 +/- 0.08 dex, and with an inferred mass of M_star = 1.03 +/- 0.03 M_sun and 
        a radius of R_star = 1.28 +/- 0.07 R_sun. We discuss the importance of K2-280b for 
        testing formation scenarios of sub-Saturn planets and the current sample of this 
        intriguing group of planets that are absent in the Solar System. 
    """
    )

    print(prediction)


if __name__ == "__main__":
    main()
