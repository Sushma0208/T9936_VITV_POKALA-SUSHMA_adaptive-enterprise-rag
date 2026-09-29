def filter_by_role(results, user_role):
    """
    Filter documents according to access type and user role.

    Public documents are accessible to all users.
    Restricted documents require the user's role to be
    present in allowed_roles.
    """

    filtered_results = []

    for result in results:

        document = result["document"]

        access_type = document.get(
            "access_type",
            "restricted"
        )

        # Public documents are accessible to everyone
        if access_type == "public":
            filtered_results.append(result)
            continue

        # Restricted documents require authorization
        allowed_roles = document.get(
            "allowed_roles",
            []
        )

        if user_role in allowed_roles:
            filtered_results.append(result)

    return filtered_results